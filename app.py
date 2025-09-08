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
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io
import re
import unicodedata

# === Impor untuk Fitur Replicate API ===
from sklearn.cluster import KMeans
import math
import replicate
import requests
# ==========================================

# === NEW IMPORTS FOR ENHANCED TEXT PROCESSING ===
from scipy import ndimage
from skimage import morphology, segmentation
from scipy.spatial.distance import euclidean
import colorsys
from collections import Counter
# === NEW: Fuzzy string matching untuk robust text detection ===
try:
    from fuzzywuzzy import fuzz, process
    print("✅ FuzzyWuzzy loaded successfully")
except ImportError:
    print("⚠️  FuzzyWuzzy not available. Install with: pip install fuzzywuzzy python-Levenshtein")
    fuzz = None
    process = None
# ================================================

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

# ===== ENHANCED TEXT PROCESSING FUNCTIONS - QUICK FIX =====

def detect_text_advanced_quick_fix(image_np, target_text):
    """
    Quick fix untuk text detection yang lebih robust tanpa library tambahan
    """
    try:
        # Normalisasi target text
        target_normalized = re.sub(r'\s+', ' ', target_text.strip().lower())
        
        print(f"🔍 Looking for: '{target_text}' (normalized: '{target_normalized}')")
        
        # Multiple preprocessing approaches
        preprocessing_methods = []
        
        # 1. Original image
        preprocessing_methods.append(("original", image_np))
        
        # 2. Grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        preprocessing_methods.append(("grayscale", gray))
        
        # 3. Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        preprocessing_methods.append(("enhanced_contrast", enhanced))
        
        # 4. Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessing_methods.append(("threshold", thresh))
        
        # 5. Scaling up untuk text kecil
        h, w = image_np.shape[:2]
        scaled_up = cv2.resize(image_np, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        preprocessing_methods.append(("scaled_2x", scaled_up))
        
        best_match = None
        best_score = 0
        all_detections = []
        
        # Test setiap preprocessing method
        for method_name, processed_img in preprocessing_methods:
            try:
                print(f"   Testing {method_name}...")
                results = ocr_reader.readtext(processed_img, detail=1)
                
                for (bbox, detected_text, confidence) in results:
                    if confidence < 0.1:  # Threshold sangat rendah
                        continue
                    
                    # Multiple similarity checks
                    detected_normalized = re.sub(r'\s+', ' ', detected_text.strip().lower())
                    
                    # Scoring system
                    scores = []
                    
                    # 1. Exact match
                    if target_normalized == detected_normalized:
                        scores.append(1.0)
                        print(f"      ✅ EXACT MATCH: '{detected_text}'")
                    
                    # 2. Contains match (kedua arah)
                    if target_normalized in detected_normalized:
                        scores.append(0.9)
                        print(f"      🎯 CONTAINS (target in detected): '{detected_text}'")
                    elif detected_normalized in target_normalized:
                        scores.append(0.85)
                        print(f"      🎯 CONTAINS (detected in target): '{detected_text}'")
                    
                    # 3. Word matching
                    target_words = set(target_normalized.split())
                    detected_words = set(detected_normalized.split())
                    common_words = target_words.intersection(detected_words)
                    
                    if len(target_words) > 0 and len(common_words) > 0:
                        word_score = len(common_words) / len(target_words)
                        if word_score > 0.5:  # Minimal 50% kata sama
                            scores.append(word_score * 0.8)
                            print(f"      📝 WORD MATCH ({word_score:.2f}): '{detected_text}'")
                    
                    # 4. Character overlap
                    target_chars = set(target_normalized.replace(' ', ''))
                    detected_chars = set(detected_normalized.replace(' ', ''))
                    if len(target_chars) > 0:
                        char_overlap = len(target_chars.intersection(detected_chars)) / len(target_chars)
                        if char_overlap > 0.6:  # Minimal 60% karakter sama
                            scores.append(char_overlap * 0.6)
                            print(f"      🔤 CHAR MATCH ({char_overlap:.2f}): '{detected_text}'")
                    
                    # 5. Simple edit distance (approximate)
                    def simple_similarity(s1, s2):
                        if len(s1) == 0 or len(s2) == 0:
                            return 0
                        matches = sum(1 for a, b in zip(s1, s2) if a == b)
                        return matches / max(len(s1), len(s2))
                    
                    edit_sim = simple_similarity(target_normalized, detected_normalized)
                    if edit_sim > 0.6:
                        scores.append(edit_sim * 0.5)
                        print(f"      ✏️ EDIT SIM ({edit_sim:.2f}): '{detected_text}'")
                    
                    # 6. FuzzyWuzzy matching (jika tersedia)
                    if fuzz:
                        fuzzy_ratio = fuzz.ratio(target_normalized, detected_normalized) / 100
                        if fuzzy_ratio > 0.6:
                            scores.append(fuzzy_ratio * 0.7)
                            print(f"      🔄 FUZZY MATCH ({fuzzy_ratio:.2f}): '{detected_text}'")
                    
                    # Calculate final score
                    if scores:
                        max_similarity = max(scores)
                        final_score = max_similarity * confidence
                        
                        detection_info = {
                            'bbox': bbox,
                            'text': detected_text,
                            'confidence': confidence,
                            'similarity': max_similarity,
                            'final_score': final_score,
                            'method': method_name
                        }
                        
                        all_detections.append(detection_info)
                        
                        print(f"      📊 Final score: {final_score:.3f} (sim: {max_similarity:.3f}, conf: {confidence:.3f})")
                        
                        if final_score > best_score:
                            best_score = final_score
                            best_match = detection_info
                
            except Exception as e:
                print(f"      ❌ Error with {method_name}: {e}")
                continue
        
        # Jika tidak ada hasil bagus, coba lagi dengan threshold lebih rendah
        if not best_match or best_score < 0.3:
            print(f"   🔄 Retrying with very low threshold...")
            
            # Ambil semua deteksi yang memiliki kata yang sama
            for detection in all_detections:
                detected_words = set(detection['text'].lower().split())
                target_words = set(target_text.lower().split())
                
                # Jika ada word yang sama, beri bonus
                common = detected_words.intersection(target_words)
                if common:
                    bonus_score = len(common) / max(len(target_words), 1) * detection['confidence']
                    if bonus_score > best_score:
                        best_score = bonus_score
                        best_match = detection
                        print(f"      🎁 BONUS MATCH: '{detection['text']}' | Common words: {common}")
        
        # Jika masih tidak ada, coba fuzzy search pada semua hasil
        if not best_match and all_detections and fuzz:
            print(f"   🔄 Trying fuzzy search on all detections...")
            all_texts = [d['text'] for d in all_detections]
            fuzzy_result = process.extractOne(target_text, all_texts, scorer=fuzz.token_sort_ratio)
            
            if fuzzy_result and fuzzy_result[1] >= 60:  # 60% similarity threshold
                matched_text = fuzzy_result[0]
                matched_detection = next(d for d in all_detections if d['text'] == matched_text)
                matched_detection['similarity'] = fuzzy_result[1] / 100
                matched_detection['final_score'] = (fuzzy_result[1] / 100) * matched_detection['confidence']
                best_match = matched_detection
                print(f"   ✅ Fuzzy match found: '{matched_text}' with {fuzzy_result[1]}% similarity")
        
        # Print summary
        if best_match:
            print(f"✅ FOUND: '{best_match['text']}' | Score: {best_score:.3f} | Method: {best_match['method']}")
            return (best_match['bbox'], best_match['text'], best_match['confidence'], best_match['final_score'])
        else:
            print(f"❌ NOT FOUND: '{target_text}'")
            print("🔍 All detected texts:")
            unique_texts = list(set([d['text'] for d in all_detections]))[:15]
            for i, text in enumerate(unique_texts):
                print(f"   {i+1:2d}. '{text}'")
            return None
            
    except Exception as e:
        print(f"❌ Error in text detection: {e}")
        return None

def extract_text_mask(image_np, bbox, padding=3):
    """
    Ekstrak mask untuk area text dengan padding
    """
    try:
        # Convert bbox to int coordinates
        points = np.array(bbox, dtype=np.int32)
        
        # Create mask
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # Add padding to mask
        kernel = np.ones((padding*2+1, padding*2+1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
        
    except Exception as e:
        print(f"Error creating text mask: {e}")
        return None

def advanced_color_extraction(image_np, bbox, mask=None):
    """
    Ekstraksi warna text yang lebih advanced
    """
    try:
        # Get bounding rectangle
        points = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        
        # Extract text area
        text_area = image_np[y:y+h, x:x+w]
        
        if mask is not None:
            mask_area = mask[y:y+h, x:x+w]
        else:
            # Create simple mask
            mask_area = np.ones(text_area.shape[:2], dtype=np.uint8) * 255
        
        # Convert to different color spaces for better analysis
        text_rgb = cv2.cvtColor(text_area, cv2.COLOR_BGR2RGB)
        text_hsv = cv2.cvtColor(text_area, cv2.COLOR_BGR2HSV)
        text_lab = cv2.cvtColor(text_area, cv2.COLOR_BGR2LAB)
        
        # Method 1: Edge-based text color detection
        edges = cv2.Canny(cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_pixels = text_rgb[edges > 0]
        
        # Method 2: Otsu thresholding to separate text from background
        gray = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract text pixels (assuming text is darker - adjust if needed)
        text_pixels_dark = text_rgb[thresh == 0]
        text_pixels_bright = text_rgb[thresh == 255]
        
        # Determine which is likely the text based on edge alignment
        if len(text_pixels_dark) > 0 and len(text_pixels_bright) > 0:
            # Use edge information to determine text color
            edge_coords = np.where(edges > 0)
            if len(edge_coords[0]) > 0:
                edge_thresh_values = thresh[edges > 0]
                dark_edges = np.sum(edge_thresh_values == 0)
                bright_edges = np.sum(edge_thresh_values == 255)
                
                if dark_edges >= bright_edges:
                    text_pixels = text_pixels_dark
                else:
                    text_pixels = text_pixels_bright
            else:
                # Fallback to smaller area (likely text)
                if len(text_pixels_dark) <= len(text_pixels_bright):
                    text_pixels = text_pixels_dark
                else:
                    text_pixels = text_pixels_bright
        else:
            text_pixels = edge_pixels if len(edge_pixels) > 0 else text_rgb.reshape((-1, 3))
        
        if len(text_pixels) == 0:
            return (0, 0, 0)
        
        # Get dominant color using KMeans
        if len(text_pixels) > 1:
            n_clusters = min(3, len(text_pixels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(text_pixels)
            
            # Get the most frequent cluster
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            most_frequent_idx = unique[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[most_frequent_idx]
        else:
            dominant_color = text_pixels[0]
        
        return tuple(np.round(dominant_color).astype(int))
        
    except Exception as e:
        print(f"Error in advanced color extraction: {e}")
        return (0, 0, 0)

def estimate_font_properties(image_np, bbox, text):
    """
    Estimasi properti font yang lebih akurat
    """
    try:
        points = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        
        # Calculate font size based on height and character count
        char_count = len(text.replace(' ', ''))  # Don't count spaces
        if char_count == 0:
            char_count = 1
        
        # Basic font size estimation
        font_height = h * 0.8  # Text typically takes 80% of bounding box height
        
        # Adjust for character width
        avg_char_width = w / char_count if char_count > 0 else w
        
        # Estimate font size (this is approximate and depends on font)
        estimated_font_size = int(font_height * 0.75)  # Convert from pixels to points (rough)
        estimated_font_size = max(8, min(estimated_font_size, 200))  # Reasonable bounds
        
        # Analyze text characteristics
        text_area = image_np[y:y+h, x:x+w]
        gray_text = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY) if len(text_area.shape) == 3 else text_area
        
        # Detect if text is bold (based on stroke width)
        edges = cv2.Canny(gray_text, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        is_bold = edge_density > 0.1  # Threshold for bold detection
        
        # Detect if text is italic (based on slant analysis)
        # This is a simplified approach
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_italic = False
        if contours:
            # Analyze the slant of major contours
            for contour in contours:
                if cv2.contourArea(contour) > 10:  # Only check significant contours
                    rect = cv2.minAreaRect(contour)
                    angle = abs(rect[2])
                    if 10 < angle < 80:  # Italic text typically has a slant
                        is_italic = True
                        break
        
        return {
            'size': estimated_font_size,
            'bold': is_bold,
            'italic': is_italic,
            'width': w,
            'height': h,
            'char_width': avg_char_width
        }
        
    except Exception as e:
        print(f"Error estimating font properties: {e}")
        return {
            'size': 20,
            'bold': False,
            'italic': False,
            'width': 100,
            'height': 30,
            'char_width': 10
        }

def advanced_background_inpainting(image_np, mask):
    """
    Background inpainting yang lebih advanced
    """
    try:
        # Convert to PIL for easier manipulation
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        # Method 1: OpenCV inpainting (fast)
        inpainted_cv = cv2.inpaint(image_np, mask, 3, cv2.INPAINT_TELEA)
        
        # Method 2: Advanced inpainting using surrounding pixels
        # Find the bounding box of the mask
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        
        # Expand the region slightly for better context
        margin = max(10, min(w, h) // 4)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image_np.shape[1], x + w + margin)
        y2 = min(image_np.shape[0], y + h + margin)
        
        # Extract the region with context
        region = image_np[y1:y2, x1:x2]
        region_mask = mask[y1:y2, x1:x2]
        
        # Apply advanced inpainting to the region
        if region.size > 0 and np.any(region_mask):
            inpainted_region = cv2.inpaint(region, region_mask, 7, cv2.INPAINT_NS)
            
            # Put the inpainted region back
            result = image_np.copy()
            result[y1:y2, x1:x2] = inpainted_region
        else:
            result = inpainted_cv
        
        # Smooth the edges
        kernel = np.ones((3, 3), np.float32) / 9
        result = cv2.filter2D(result, -1, kernel)
        
        return result
        
    except Exception as e:
        print(f"Error in background inpainting: {e}")
        return image_np

def find_best_font(font_properties, text):
    """
    Mencari font terbaik berdasarkan properti yang terdeteksi
    """
    # Extended font list with more options
    font_candidates = [
        # Windows fonts
        ("arial.ttf", "Arial", False, False),
        ("arialbd.ttf", "Arial", True, False),
        ("ariali.ttf", "Arial", False, True),
        ("arialbi.ttf", "Arial", True, True),
        ("times.ttf", "Times New Roman", False, False),
        ("timesbd.ttf", "Times New Roman", True, False),
        ("timesi.ttf", "Times New Roman", False, True),
        ("timesbi.ttf", "Times New Roman", True, True),
        ("calibri.ttf", "Calibri", False, False),
        ("calibrib.ttf", "Calibri", True, False),
        ("calibrii.ttf", "Calibri", False, True),
        ("calibriz.ttf", "Calibri", True, True),
        ("verdana.ttf", "Verdana", False, False),
        ("verdanab.ttf", "Verdana", True, False),
        ("verdanai.ttf", "Verdana", False, True),
        ("verdanaz.ttf", "Verdana", True, True),
        ("tahoma.ttf", "Tahoma", False, False),
        ("tahomabd.ttf", "Tahoma", True, False),
        ("trebuc.ttf", "Trebuchet MS", False, False),
        ("trebucbd.ttf", "Trebuchet MS", True, False),
        ("trebucit.ttf", "Trebuchet MS", False, True),
        ("trebucbi.ttf", "Trebuchet MS", True, True),
        ("comic.ttf", "Comic Sans MS", False, False),
        ("comicbd.ttf", "Comic Sans MS", True, False),
        ("impact.ttf", "Impact", False, False),
        # System fonts
        ("Arial.ttf", "Arial", False, False),
        ("Times.ttf", "Times", False, False),
        ("Helvetica.ttf", "Helvetica", False, False),
    ]
    
    font_directories = [
        "C:/Windows/Fonts/",
        "C:/WINDOWS/Fonts/", 
        "/System/Library/Fonts/",  # macOS
        "/usr/share/fonts/truetype/",  # Linux
        "/usr/share/fonts/TTF/",
        ""  # Current directory
    ]
    
    target_bold = font_properties.get('bold', False)
    target_italic = font_properties.get('italic', False)
    target_size = font_properties.get('size', 20)
    
    best_font = None
    best_score = -1
    
    for font_file, font_family, is_bold, is_italic in font_candidates:
        # Score based on style matching
        style_score = 0
        if is_bold == target_bold:
            style_score += 1
        if is_italic == target_italic:
            style_score += 1
            
        if style_score < best_score:
            continue
            
        # Try to load the font
        for directory in font_directories:
            font_path = os.path.join(directory, font_file)
            if os.path.exists(font_path):
                try:
                    # Test the font with target size
                    test_font = ImageFont.truetype(font_path, target_size)
                    
                    # Quick test - try to measure text
                    dummy_img = Image.new('RGB', (100, 100), (255, 255, 255))
                    dummy_draw = ImageDraw.Draw(dummy_img)
                    
                    # Check if font can render the text
                    bbox = dummy_draw.textbbox((0, 0), text, font=test_font)
                    
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bounding box
                        if style_score > best_score:
                            best_font = test_font
                            best_score = style_score
                            print(f"Selected font: {font_path} (score: {style_score})")
                        break
                        
                except Exception as e:
                    continue
    
    # Fallback to default font
    if best_font is None:
        try:
            best_font = ImageFont.load_default()
            print("Using default font as fallback")
        except:
            # Ultimate fallback
            best_font = None
            print("Could not load any font")
    
    return best_font

def render_text_with_properties(image_pil, text, bbox, text_color, font_properties):
    """
    Render text dengan properti yang sesuai
    """
    try:
        # Get font
        font = find_best_font(font_properties, text)
        if font is None:
            font = ImageFont.load_default()
        
        # Calculate position
        points = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        
        # Create a drawing context
        draw = ImageDraw.Draw(image_pil)
        
        # Calculate text size with current font
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Adjust font size if text doesn't fit
        original_size = font_properties.get('size', 20)
        current_size = original_size
        
        # Scale down if text is too large
        while (text_width > w * 0.95 or text_height > h * 0.95) and current_size > 8:
            current_size = int(current_size * 0.9)
            try:
                font = find_best_font({**font_properties, 'size': current_size}, text)
                if font is None:
                    font = ImageFont.load_default()
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                break
        
        # Scale up if text is much smaller
        while (text_width < w * 0.7 and text_height < h * 0.7) and current_size < original_size * 1.5:
            test_size = int(current_size * 1.1)
            try:
                test_font = find_best_font({**font_properties, 'size': test_size}, text)
                if test_font is None:
                    break
                test_bbox = draw.textbbox((0, 0), text, font=test_font)
                test_width = test_bbox[2] - test_bbox[0]
                test_height = test_bbox[3] - test_bbox[1]
                
                if test_width <= w * 0.95 and test_height <= h * 0.95:
                    current_size = test_size
                    font = test_font
                    text_width = test_width
                    text_height = test_height
                else:
                    break
            except:
                break
        
        # Center the text in the bounding box
        text_x = x + (w - text_width) // 2
        text_y = y + (h - text_height) // 2
        
        # Draw the text
        draw.text((text_x, text_y), text, fill=text_color, font=font)
        
        return image_pil
        
    except Exception as e:
        print(f"Error rendering text: {e}")
        return image_pil

def process_text_swap_advanced_quick_fix(image_pil, original_text, new_text):
    """
    Quick fix untuk text swap dengan detection yang lebih robust
    """
    try:
        print("🚀 Starting QUICK FIX text swap process...")
        
        # Convert PIL to numpy for OCR
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 1. Quick fix text detection
        print(f"Detecting text: '{original_text}'...")
        detection_result = detect_text_advanced_quick_fix(image_bgr, original_text)
        
        if detection_result is None:
            # Coba dengan part dari text
            words = original_text.split()
            if len(words) > 1:
                print("🔄 Trying with individual words...")
                for word in words:
                    if len(word) > 2:  # Skip kata pendek
                        print(f"   Trying word: '{word}'")
                        detection_result = detect_text_advanced_quick_fix(image_bgr, word)
                        if detection_result:
                            print(f"   ✅ Found using word: '{word}'")
                            break
        
        if detection_result is None:
            available_texts = []
            try:
                results = ocr_reader.readtext(image_bgr, detail=1)
                available_texts = [text for (_, text, conf) in results if conf > 0.1]
            except:
                pass
            
            suggestion_msg = f"Text '{original_text}' tidak ditemukan. "
            if available_texts:
                suggestion_msg += f"Text yang terdeteksi: {', '.join(available_texts[:10])}"
            else:
                suggestion_msg += "Pastikan gambar memiliki kualitas yang baik dan text terlihat jelas."
            
            raise ValueError(suggestion_msg)
        
        bbox, detected_text, confidence, score = detection_result
        print(f"✅ Text detected: '{detected_text}' with confidence {confidence:.2f} (score: {score:.2f})")
        
        # 2. Create text mask
        print("Creating text mask...")
        text_mask = extract_text_mask(image_bgr, bbox, padding=2)
        
        if text_mask is None:
            raise ValueError("Failed to create text mask")
        
        # 3. Extract text color
        print("Extracting text color...")
        text_color = advanced_color_extraction(image_bgr, bbox, text_mask)
        print(f"Detected text color: {text_color}")
        
        # 4. Estimate font properties
        print("Estimating font properties...")
        font_props = estimate_font_properties(image_bgr, bbox, detected_text)
        print(f"Font properties: {font_props}")
        
        # 5. Advanced background inpainting
        print("Performing background inpainting...")
        inpainted_bgr = advanced_background_inpainting(image_bgr, text_mask)
        
        # Convert back to PIL RGB
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(inpainted_rgb)
        
        # 6. Render new text
        print(f"Rendering new text: '{new_text}'...")
        final_image = render_text_with_properties(
            result_image, 
            new_text, 
            bbox, 
            text_color, 
            font_props
        )
        
        return final_image, detected_text, confidence
        
    except Exception as e:
        print(f"❌ Error in quick fix text swap: {e}")
        raise e

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

# --- 7. ENHANCED Text Swap Process - QUICK FIX VERSION ---
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
        # Reset file pointer and read image
        image_file.seek(0)
        image = Image.open(image_file).convert('RGB')
        
        print(f"Processing text swap: '{original_text}' -> '{new_text}'")
        
        # Use the QUICK FIX text swap function
        result_image, detected_text, confidence = process_text_swap_advanced_quick_fix(
            image, original_text, new_text
        )
        
        # Save result
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        unique_id = str(uuid.uuid4())
        result_filename = f"{unique_id}_text_swap.jpg"
        result_path = os.path.join(upload_folder, result_filename)
        
        # Save with high quality
        result_image.save(result_path, 'JPEG', quality=95, optimize=True)
        
        # Save to database
        relative_path = os.path.join('uploads', result_filename)
        new_text_swap = SwapResult(result_image_path=relative_path, author=current_user)
        db.session.add(new_text_swap)
        db.session.commit()
        
        # Convert to base64 for response
        buffer = io.BytesIO()
        result_image.save(buffer, format='JPEG', quality=95, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        result_base64 = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({
            'result_image': result_base64,
            'detected_text': detected_text,
            'confidence': f"{confidence:.2f}",
            'success': True,
            'message': f"✅ Text '{original_text}' berhasil diganti dengan '{new_text}'"
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error dalam text swap: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Enhanced error suggestions
        suggestions = []
        debug_info = ""
        
        if "tidak ditemukan" in error_msg.lower() or "not found" in error_msg.lower():
            suggestions = [
                "🔍 Pastikan text yang dicari benar-benar ada dan terlihat jelas di gambar",
                "🔍 Coba gunakan kata kunci yang lebih spesifik atau unik",
                "🔤 Coba variasi kapitalisasi: 'Jepang', 'JEPANG', atau 'jepang'",
                "✂️ Coba gunakan bagian dari text (misal: 'Jepang' dari 'di Jepang')",
                "🖼️ Pastikan gambar memiliki kualitas HD dan text tidak terpotong"
            ]
            
            # Extract available texts for debugging
            if "Text yang terdeteksi:" in error_msg:
                debug_info = "💡 Text yang terdeteksi di gambar: lihat pesan error di atas"
            
        elif "mask" in error_msg.lower():
            suggestions = [
                "🎯 Text berhasil ditemukan tapi gagal membuat mask",
                "🔄 Coba dengan gambar yang berbeda atau crop area text saja",
                "🔍 Pastikan text tidak terlalu kecil atau terdistorsi"
            ]
        
        else:
            suggestions = [
                "🔧 Terjadi kesalahan teknis dalam pemrosesan",
                "📱 Coba refresh halaman dan upload ulang",
                "🖼️ Pastikan format gambar adalah JPG/PNG yang valid"
            ]
        
        error_response = {
            'error': error_msg,
            'suggestions': suggestions,
            'debug_tip': "Untuk hasil terbaik, gunakan gambar HD dengan text yang kontras dan jelas"
        }
        
        if debug_info:
            error_response['debug_info'] = debug_info
        
        return jsonify(error_response), 500

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

# --- 10. Testing Function (Opsional untuk debugging) ---
def test_text_detection_manual(image_path, target_text):
    """
    Fungsi untuk testing manual dari command line
    """
    print("="*60)
    print(f"TESTING TEXT DETECTION")
    print(f"Image: {image_path}")
    print(f"Target: '{target_text}'")
    print("="*60)
    
    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    # Test detection
    result = detect_text_advanced_quick_fix(image_bgr, target_text)
    
    if result:
        bbox, detected_text, confidence, score = result
        print(f"\n🎉 SUCCESS!")
        print(f"   Detected: '{detected_text}'")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Score: {score:.3f}")
        
        # Visualize result (optional)
        try:
            # Draw bounding box
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(image_bgr, [points], True, (0, 255, 0), 2)
            
            # Add text label
            x, y, w, h = cv2.boundingRect(points)
            cv2.putText(image_bgr, f"Found: {detected_text}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save result
            output_path = f"debug_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, image_bgr)
            print(f"   Debug image saved: {output_path}")
            
        except Exception as e:
            print(f"   (Visualization failed: {e})")
        
    else:
        print(f"\n❌ FAILED to detect '{target_text}'")
    
    print("="*60)

# --- 11. Menjalankan Aplikasi & Membuat Database ---
if __name__ == '__main__':
    # Uncomment untuk testing manual
    # test_text_detection_manual("path/to/test/image.jpg", "Jepang")
    
    # Jalankan app
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)