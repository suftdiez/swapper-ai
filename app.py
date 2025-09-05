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

load_dotenv()
app.config['GOOGLE_CLIENT_ID'] = os.getenv("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv("GOOGLE_CLIENT_SECRET")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

# === Impor Baru untuk Fitur Replicate API ===
import replicate
import requests
# ==========================================

# --- 1. Konfigurasi Aplikasi & Database ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = '8f4f917c2a7f4a2d8a9b1c7d2e4f6a8b'

# --- Konfigurasi untuk Google OAuth ---
app.config['GOOGLE_CLIENT_ID'] = os.getenv("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv("GOOGLE_CLIENT_SECRET")

# === Konfigurasi untuk Replicate API ===
# PENTING: Tempel (paste) API token Anda di sini
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

# --- 2. Model Database & Konfigurasi Login (Tetap Sama) ---
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

# --- 3. Logika Inti AI (Tanpa GFPGAN) ---
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

def process_image(file_storage):
    in_memory_file = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

# --- 4. Rute Halaman (Frontend) (Tetap Sama) ---
@app.route("/")
def home(): return render_template('landing.html', title="Selamat Datang")
# ... (Semua rute halaman lain tetap sama) ...
@app.route("/app")
@login_required
def face_swapper_app(): return render_template('index.html', title="AI Face Swapper")
@app.route("/profile")
@login_required
def profile(): return render_template('profile.html', title='Profil Saya')
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

# --- 5. Rute Autentikasi (Tetap Sama) ---
# ... (Semua rute autentikasi Anda tetap sama) ...
@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('face_swapper_app'))
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
    if current_user.is_authenticated: return redirect(url_for('face_swapper_app'))
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

# --- 6. Rute Login Google (Tetap Sama) ---
# ... (Semua rute Google login Anda tetap sama) ...
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

    @app.route("/text-swap")
@login_required
def text_swapper_app():
    return render_template('text_swapper.html', title="AI Text Swapper")

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
        results = ocr_reader.readtext(image_np)
        
        # Cari text yang cocok
        text_found = False
        for (bbox, text, confidence) in results:
            if confidence > 0.5 and original_text.lower() in text.lower():
                text_found = True
                # Hapus text lama dengan menggambar kotak putih
                top_left = tuple([int(val) for val in bbox[0]])
                bottom_right = tuple([int(val) for val in bbox[2]])
                
                draw = ImageDraw.Draw(image)
                draw.rectangle([top_left, bottom_right], fill='white')
                
                # Tambahkan text baru
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                draw.text(top_left, new_text, fill='black', font=font)
                break
        
        if not text_found:
            return jsonify({'error': f'Text "{original_text}" tidak ditemukan dalam gambar'}), 400
        
        # Simpan hasil
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        unique_id = str(uuid.uuid4())
        result_filename = f"{unique_id}_text_swap.jpg"
        result_path = os.path.join(upload_folder, result_filename)
        
        image.save(result_path, 'JPEG')
        
        # Simpan ke database
        relative_path = os.path.join('uploads', result_filename)
        new_text_swap = SwapResult(result_image_path=relative_path, author=current_user)
        db.session.add(new_text_swap)
        db.session.commit()
        
        # Convert ke base64 untuk response
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        result_base64 = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({'result_image': result_base64})
        
    except Exception as e:
        print(f"Error saat text swap: {e}")
        return jsonify({'error': 'Terjadi kesalahan saat memproses gambar'}), 500

# --- 7. Rute API untuk Face Swap (DIPERBARUI DENGAN Real-ESRGAN) ---
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

# --- Rute Hapus Riwayat (Tetap Sama) ---
@app.route("/delete_swap/<int:swap_id>", methods=['POST'])
@login_required
def delete_swap(swap_id):
    # ... (kode hapus Anda tetap sama) ...
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

# --- 8. Menjalankan Aplikasi & Membuat Database ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
