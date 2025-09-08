import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
from authlib.integrations.flask_client import OAuth
from datetime import datetime, timedelta
from dotenv import load_dotenv
import base64
import uuid
import easyocr
from PIL import Image, ImageDraw, ImageFont
import io

# === Impor untuk Fitur Replicate API ===
from sklearn.cluster import KMeans
import math
import replicate
import requests
# ==========================================

load_dotenv()

# --- 1. Konfigurasi Aplikasi & Database ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = '8f4f917c2a7f4a2d8a9b1c7d2e4f6a8b'

# --- Konfigurasi untuk Google OAuth ---
app.config['GOOGLE_CLIENT_ID'] = os.getenv("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv("GOOGLE_CLIENT_SECRET")

# === Konfigurasi untuk Replicate API ===
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
# ======================================

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
oauth = OAuth(app)

@app.context_processor
def inject_timedelta():
    return dict(timedelta=timedelta)

# --- 2. Model Database & Konfigurasi Login ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)
    swaps = db.relationship('SwapResult', backref='author', lazy=True)

class SwapResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result_image_path = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 3. Logika Inti AI ---
face_analyzer = None
swapper = None
try:
    print("Memuat model Face Analysis...")
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("Model Face Analysis siap.")

    print("Memuat model Face Swapper...")
    model_path = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'inswapper_128', 'inswapper_128.onnx')
    swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=True, providers=['CPUExecutionProvider'])
    print("Swapper siap.")
except Exception as e:
    print(f"!!! PERINGATAN: GAGAL MEMUAT MODEL AI LOKAL: {e}")

# --- Inisialisasi OCR Reader ---
ocr_reader = None
try:
    print("Memuat OCR Reader...")
    ocr_reader = easyocr.Reader(['en', 'id'])  # English dan Indonesian
    print("OCR Reader siap.")
except Exception as e:
    print(f"!!! PERINGATAN: GAGAL MEMUAT OCR READER: {e}")

# === PERBAIKAN INDENTASI - COPY PASTE KODE INI ===
# Pastikan tidak ada tab, hanya gunakan 4 spasi untuk indentasi

def analyze_text_properties(image_np, bbox, text, confidence):
    """
    Analisis properti teks yang terdeteksi untuk matching yang lebih baik
    """
    try:
        # Konversi bbox ke koordinat integer
        top_left = [int(val) for val in bbox[0]]
        bottom_right = [int(val) for val in bbox[2]]
        
        # Ekstrak area teks
        text_area = image_np[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        if text_area.size == 0:
            return None
            
        # Hitung ukuran teks
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        
        # Estimasi font size berdasarkan tinggi
        estimated_font_size = max(12, int(height * 0.7))
        
        # Analisis warna dominan di area teks
        text_colors = analyze_text_color(text_area)
        background_color = analyze_background_color(image_np, bbox)
        
        return {
            'bbox': bbox,
            'width': width,
            'height': height,
            'estimated_font_size': estimated_font_size,
            'text_color': text_colors['dominant'],
            'background_color': background_color,
            'top_left': top_left,
            'bottom_right': bottom_right
        }
        
    except Exception as e:
        print(f"Error analyzing text properties: {e}")
        return None


def analyze_text_color(text_area):
    """
    Analisis warna teks dominan
    """
    try:
        # Konversi ke RGB jika perlu
        if len(text_area.shape) == 3:
            text_area_rgb = cv2.cvtColor(text_area, cv2.COLOR_BGR2RGB)
        else:
            text_area_rgb = text_area
            
        # Reshape untuk clustering
        pixels = text_area_rgb.reshape((-1, 3))
        
        # Gunakan KMeans untuk find warna dominan
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Hitung frekuensi setiap cluster
        unique, counts = np.unique(labels, return_counts=True)
        
        # Pilih warna yang lebih kontras (biasanya teks)
        color1, color2 = colors[0], colors[1]
        
        # Pilih warna yang lebih gelap/terang sebagai teks
        brightness1 = np.mean(color1)
        brightness2 = np.mean(color2)
        
        if brightness1 < brightness2:
            text_color = tuple(color1)
        else:
            text_color = tuple(color2)
            
        return {
            'dominant': text_color,
            'colors': colors
        }
        
    except Exception as e:
        print(f"Error analyzing text color: {e}")
        return {'dominant': (0, 0, 0), 'colors': [(0, 0, 0)]}


def analyze_background_color(image_np, bbox):
    """
    Analisis warna background di sekitar teks
    """
    try:
        top_left = [int(val) for val in bbox[0]]
        bottom_right = [int(val) for val in bbox[2]]
        
        # Expand area sedikit untuk capture background
        margin = 10
        expanded_top = max(0, top_left[1] - margin)
        expanded_bottom = min(image_np.shape[0], bottom_right[1] + margin)
        expanded_left = max(0, top_left[0] - margin)
        expanded_right = min(image_np.shape[1], bottom_right[0] + margin)
        
        # Ambil area yang diperluas
        expanded_area = image_np[expanded_top:expanded_bottom, expanded_left:expanded_right]
        
        if expanded_area.size == 0:
            return (255, 255, 255)
            
        # Konversi ke RGB
        if len(expanded_area.shape) == 3:
            expanded_area_rgb = cv2.cvtColor(expanded_area, cv2.COLOR_BGR2RGB)
        else:
            expanded_area_rgb = expanded_area
            
        # Ambil pixel dari border area (bukan tengah yang merupakan teks)
        border_pixels = []
        h, w = expanded_area_rgb.shape[:2]
        
        # Top and bottom borders
        border_pixels.extend(expanded_area_rgb[0:margin].reshape((-1, 3)))
        border_pixels.extend(expanded_area_rgb[h-margin:h].reshape((-1, 3)))
        
        # Left and right borders  
        border_pixels.extend(expanded_area_rgb[:, 0:margin].reshape((-1, 3)))
        border_pixels.extend(expanded_area_rgb[:, w-margin:w].reshape((-1, 3)))
        
        if len(border_pixels) > 0:
            border_pixels = np.array(border_pixels)
            background_color = tuple(np.mean(border_pixels, axis=0).astype(int))
        else:
            # Fallback ke warna rata-rata
            background_color = tuple(np.mean(expanded_area_rgb, axis=(0, 1)).astype(int))
            
        return background_color
        
    except Exception as e:
        print(f"Error analyzing background color: {e}")
        return (255, 255, 255)


def create_better_text_replacement(image, text_props, new_text):
    """
    Buat replacement teks yang lebih baik dengan properties matching
    """
    try:
        # Buat background patch yang seamless
        background_patch = create_seamless_background(image, text_props)
        
        # Tempel background patch
        image.paste(background_patch, (text_props['top_left'][0], text_props['top_left'][1]))
        
        # Cari font yang sesuai
        font = find_best_matching_font(text_props['estimated_font_size'], new_text, 
                                     text_props['width'], text_props['height'])
        
        # Gambar teks baru
        draw = ImageDraw.Draw(image)
        
        # Hitung posisi untuk centering
        text_bbox = draw.textbbox((0, 0), new_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Posisi teks di tengah area
        x = text_props['top_left'][0] + (text_props['width'] - text_width) // 2
        y = text_props['top_left'][1] + (text_props['height'] - text_height) // 2
        
        # Gambar teks dengan warna yang sesuai
        draw.text((x, y), new_text, fill=text_props['text_color'], font=font)
        
        return image
        
    except Exception as e:
        print(f"Error creating text replacement: {e}")
        return image


def create_seamless_background(image, text_props):
    """
    Buat background yang seamless untuk mengganti area teks
    """
    try:
        # Ambil area sekitar teks untuk sampling
        margin = 5
        x1, y1 = text_props['top_left']
        x2, y2 = text_props['bottom_right']
        
        # Expand untuk sampling
        sample_x1 = max(0, x1 - margin * 2)
        sample_y1 = max(0, y1 - margin * 2)
        sample_x2 = min(image.width, x2 + margin * 2)
        sample_y2 = min(image.height, y2 + margin * 2)
        
        # Crop area sampling
        sample_area = image.crop((sample_x1, sample_y1, sample_x2, sample_y2))
        
        # Buat patch dengan ukuran text area
        patch_width = x2 - x1
        patch_height = y2 - y1
        
        # Method 1: Gunakan warna background dominan
        background_patch = Image.new('RGB', (patch_width, patch_height), 
                                   text_props['background_color'])
        
        # Method 2: Jika memungkinkan, gunakan content-aware fill sederhana
        # dengan blur dan resample dari area sekitar
        if sample_area.width > 10 and sample_area.height > 10:
            # Resize sample area ke ukuran patch
            resized_sample = sample_area.resize((patch_width, patch_height), 
                                              Image.LANCZOS)
            
            # Blur untuk mengurangi detail
            from PIL import ImageEnhance
            blurred = resized_sample.filter(ImageEnhance.Color(resized_sample).enhance(0.8))
            
            # Blend dengan background color
            background_patch = Image.blend(background_patch, blurred, 0.3)
        
        return background_patch
        
    except Exception as e:
        print(f"Error creating seamless background: {e}")
        # Fallback ke warna solid
        return Image.new('RGB', (text_props['width'], text_props['height']), 
                        text_props['background_color'])


def find_best_matching_font(target_size, text, max_width, max_height):
    """
    Cari font yang paling sesuai dengan ukuran target
    """
    # Daftar font yang umum tersedia
    font_paths = [
        "arial.ttf", "Arial.ttf",
        "calibri.ttf", "Calibri.ttf", 
        "times.ttf", "Times.ttf",
        "verdana.ttf", "Verdana.ttf",
        "tahoma.ttf", "Tahoma.ttf",
        "trebuc.ttf", "Trebuchet.ttf",
        "comic.ttf", "Comic.ttf",
        "impact.ttf", "Impact.ttf"
    ]
    
    # Coba berbagai lokasi font Windows
    font_directories = [
        "C:/Windows/Fonts/",
        "C:/WINDOWS/Fonts/",
        "/System/Library/Fonts/",  # macOS
        "/usr/share/fonts/",       # Linux
        ""  # Current directory / default
    ]
    
    best_font = None
    best_size = target_size
    
    # Coba find font yang tersedia
    for font_name in font_paths:
        for font_dir in font_directories:
            try:
                font_path = os.path.join(font_dir, font_name)
                if os.path.exists(font_path):
                    # Test dengan ukuran target
                    test_font = ImageFont.truetype(font_path, target_size)
                    
                    # Buat dummy image untuk test ukuran text
                    dummy_img = Image.new('RGB', (1, 1))
                    dummy_draw = ImageDraw.Draw(dummy_img)
                    
                    # Cek ukuran text dengan font ini
                    bbox = dummy_draw.textbbox((0, 0), text, font=test_font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Adjust font size jika terlalu besar
                    if text_width > max_width or text_height > max_height:
                        # Hitung scaling factor
                        scale_w = max_width / text_width if text_width > 0 else 1
                        scale_h = max_height / text_height if text_height > 0 else 1
                        scale = min(scale_w, scale_h, 1.0)
                        
                        adjusted_size = int(target_size * scale * 0.8)  # 80% untuk safety margin
                        adjusted_size = max(8, adjusted_size)  # Minimum size
                        
                        best_font = ImageFont.truetype(font_path, adjusted_size)
                        best_size = adjusted_size
                    else:
                        best_font = test_font
                        best_size = target_size
                    
                    # Jika sudah dapat font yang bagus, stop searching
                    if best_font:
                        print(f"Using font: {font_path} with size {best_size}")
                        return best_font
                        
            except Exception as e:
                continue
    
    # Fallback ke default font jika tidak ada yang cocok
    try:
        # Adjust size untuk default font
        adjusted_size = min(target_size, max_width // len(text) if len(text) > 0 else target_size)
        adjusted_size = max(8, adjusted_size)
        best_font = ImageFont.load_default()
        print(f"Using default font with estimated size {adjusted_size}")
    except:
        best_font = ImageFont.load_default()
        print("Using basic default font")
    
    return best_font

def process_image(file_storage):
    in_memory_file = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

# --- 4. Rute Halaman (Frontend) ---
@app.route("/")
def home(): 
    return render_template('landing.html', title="Selamat Datang")

@app.route("/app")
@login_required
def face_swapper_app(): 
    return render_template('index.html', title="AI Face Swapper")

@app.route("/text-swap")
@login_required
def text_swapper_app():
    return render_template('text_swapper.html', title="AI Text Swapper")

@app.route("/profile")
@login_required
def profile(): 
    return render_template('profile.html', title='Profil Saya')

@app.route("/gallery")
@login_required
def gallery():
    results = SwapResult.query.filter_by(author=current_user).order_by(SwapResult.timestamp.desc()).all()
    return render_template('gallery.html', title='Galeri Saya', results=results)

@app.route("/about")
def about():
    return render_template('about.html', title='Tentang Kami')

@app.route("/contack")
def contact():
    return render_template('contack.html', title='Hubungi Kami')

@app.route("/faq")
def faq():
    return render_template('faq.html', title='FAQ')

# --- 5. Rute Autentikasi ---
@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: 
        return redirect(url_for('face_swapper_app'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email sudah terdaftar. Silakan login.', 'warning')
            return redirect(url_for('login'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(email=email, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Akun berhasil dibuat! Silakan login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: 
        return redirect(url_for('face_swapper_app'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.password_hash and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('face_swapper_app'))
        else:
            flash('Login gagal. Periksa kembali email dan password Anda.', 'danger')
    return render_template('login.html', title='Login')

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

# --- 6. Rute Login Google ---
@app.route('/google/')
def google_login():
    oauth.register(name='google', client_id=app.config["GOOGLE_CLIENT_ID"], client_secret=app.config["GOOGLE_CLIENT_SECRET"], server_metadata_url='https://accounts.google.com/.well-known/openid-configuration', client_kwargs={'scope': 'openid email profile'})
    redirect_uri = url_for('google_auth', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/google/auth')
def google_auth():
    token = oauth.google.authorize_access_token()
    user_info = token.get('userinfo')
    user = User.query.filter_by(email=user_info['email']).first()
    if not user:
        new_user = User(email=user_info['email'], password_hash=None)
        db.session.add(new_user)
        db.session.commit()
        user = new_user
    login_user(user)
    return redirect(url_for('face_swapper_app'))

# --- 7. Rute Text Swap Process ---
@app.route('/text-swap-process', methods=['POST'])
@login_required
def text_swap_process():
    if not ocr_reader:
        return jsonify({'error': 'OCR Reader tidak berhasil dimuat di server.'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'Gambar dibutuhkan'}), 400
    
    image_file = request.files['image']
    original_text = request.form.get('original_text', '').strip()
    new_text = request.form.get('new_text', '').strip()
    
    if not original_text or not new_text:
        return jsonify({'error': 'Text asli dan text pengganti dibutuhkan'}), 400
    
    try:
        # Baca gambar
        image_file.seek(0)
        image = Image.open(image_file).convert('RGB')
        image_np = np.array(image)
        
        # Deteksi text dengan OCR
        results = ocr_reader.readtext(image_np, detail=1)
        
        # Cari text yang cocok dan analisis properties
        text_found = False
        best_match = None
        best_confidence = 0
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Lower threshold untuk lebih fleksibel
                # Cek kecocokan text (case insensitive dan partial matching)
                if (original_text.lower() in text.lower() or 
                    text.lower() in original_text.lower() or
                    any(word.lower() in text.lower() for word in original_text.split()) or
                    any(word.lower() in original_text.lower() for word in text.split())):
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (bbox, text, confidence)
                        text_found = True
        
        if not text_found:
            return jsonify({'error': f'Text "{original_text}" tidak ditemukan dalam gambar. Coba dengan text yang lebih spesifik.'}), 400
        
        # Analisis properties dari text yang ditemukan
        bbox, detected_text, confidence = best_match
        text_props = analyze_text_properties(image_np, bbox, detected_text, confidence)
        
        if not text_props:
            return jsonify({'error': 'Tidak dapat menganalisis properti text'}), 400
        
        print(f"Detected text: '{detected_text}' with confidence {confidence:.2f}")
        print(f"Text properties: {text_props}")
        
        # Buat replacement yang lebih baik
        result_image = create_better_text_replacement(image, text_props, new_text)
        
        # Simpan hasil
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        unique_id = str(uuid.uuid4())
        result_filename = f"{unique_id}_text_swap.jpg"
        result_path = os.path.join(upload_folder, result_filename)
        
        # Simpan dengan kualitas tinggi
        result_image.save(result_path, 'JPEG', quality=95)
        
        # Simpan ke database
        relative_path = os.path.join('uploads', result_filename)
        new_text_swap = SwapResult(result_image_path=relative_path, author=current_user)
        db.session.add(new_text_swap)
        db.session.commit()
        
        # Convert ke base64 untuk response
        buffer = io.BytesIO()
        result_image.save(buffer, format='JPEG', quality=95)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        result_base64 = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({
            'result_image': result_base64,
            'detected_text': detected_text,
            'confidence': f"{confidence:.2f}"
        })
        
    except Exception as e:
        print(f"Error saat text swap: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500

# --- 8. Rute API untuk Face Swap ---
@app.route('/swap', methods=['POST'])
@login_required
def swap_faces():
    if not face_analyzer or not swapper:
        return jsonify({'error': 'Model AI lokal tidak berhasil dimuat di server.'}), 500
    if 'source_image' not in request.files or 'target_image' not in request.files:
        return jsonify({'error': 'Gambar sumber dan target dibutuhkan'}), 400

    source_file = request.files['source_image']
    target_file = request.files['target_image']
    enhance_quality = request.form.get('enhance_quality') == 'true'

    try:
        source_file.seek(0)
        target_file.seek(0)
        source_img = process_image(source_file)
        target_img = process_image(target_file)
        source_faces = face_analyzer.get(source_img)
        target_faces = face_analyzer.get(target_img)

        if not source_faces or not target_faces:
            return jsonify({'error': 'Wajah tidak terdeteksi di salah satu gambar'}), 400
        
        source_face = source_faces[0]
        target_face = max(target_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        result_img = swapper.get(target_img, target_face, source_face, paste_back=True)

        # === JALANKAN PROSES UPSCALING VIA REPLICATE API JIKA DIMINTA ===
        if enhance_quality:
            print("Mengirim gambar ke Replicate API untuk upscaling dengan Real-ESRGAN...")
            # Ubah gambar ke format base64 untuk dikirim
            _, buffer = cv2.imencode('.jpg', result_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{img_base64}"

            # Panggil model Real-ESRGAN di Replicate dengan fitur face_enhance
            output_url = replicate.run(
                "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5237799434521bf07b14617d01091ea1e4518",
                input={
                    "image": data_uri,
                    "scale": 2,
                    "face_enhance": True # <-- KUNCI UTAMA UNTUK HASIL TERBAIK
                }
            )
            
            # Unduh gambar hasil dari URL yang diberikan Replicate
            response = requests.get(output_url)
            if response.status_code == 200:
                result_img_data = np.frombuffer(response.content, np.uint8)
                result_img = cv2.imdecode(result_img_data, cv2.IMREAD_COLOR)
            else:
                print("Gagal mengunduh gambar hasil dari Replicate.")
        # ===============================================================
        
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        unique_id = str(uuid.uuid4())
        result_filename = f"{unique_id}_result.jpg"
        result_path = os.path.join(upload_folder, result_filename)
        cv2.imwrite(result_path, result_img)
        
        relative_path = os.path.join('uploads', result_filename)
        new_swap = SwapResult(result_image_path=relative_path, author=current_user)
        db.session.add(new_swap)
        db.session.commit()
        
        _, buffer = cv2.imencode('.jpg', result_img)
        result_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        return jsonify({'result_image': result_base64})

    except Exception as e:
        print(f"Terjadi error saat swap: {e}")
        return jsonify({'error': 'Terjadi kesalahan internal saat memproses gambar'}), 500

# --- 9. Rute Hapus Riwayat ---
@app.route("/delete_swap/<int:swap_id>", methods=['POST'])
@login_required
def delete_swap(swap_id):
    swap_to_delete = SwapResult.query.get_or_404(swap_id)
    if swap_to_delete.author != current_user:
        flash('Anda tidak memiliki izin untuk menghapus item ini.', 'danger')
        return redirect(url_for('gallery'))
    try:
        image_path = os.path.join(app.root_path, 'static', swap_to_delete.result_image_path.replace('/', os.sep))
        if os.path.exists(image_path):
            os.remove(image_path)
        db.session.delete(swap_to_delete)
        db.session.commit()
        flash('Gambar berhasil dihapus dari riwayat.', 'success')
    except Exception as e:
        print(f"Error saat menghapus: {e}")
        flash('Terjadi kesalahan saat mencoba menghapus gambar.', 'danger')
        db.session.rollback()
    return redirect(url_for('gallery'))

# --- 10. Menjalankan Aplikasi & Membuat Database ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)