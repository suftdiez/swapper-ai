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

# === FIXED IMPORTS - Mengatasi ANTIALIAS deprecation ===
from scipy import ndimage
from skimage import morphology, segmentation
from scipy.spatial.distance import euclidean
import colorsys
from collections import Counter

# Fix untuk ANTIALIAS yang deprecated di Pillow baru
try:
    from PIL import Image
    RESAMPLE_FILTER = Image.LANCZOS  # Gunakan LANCZOS sebagai pengganti ANTIALIAS
except AttributeError:
    RESAMPLE_FILTER = Image.ANTIALIAS

# Enhanced fuzzy matching untuk robust text detection
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

# --- Enhanced OCR Reader dengan Error Handling yang Lebih Baik ---
ocr_reader = None
try:
    print("Memuat Enhanced OCR Reader...")
    # Load dengan lebih banyak bahasa untuk deteksi yang lebih baik
    ocr_reader = easyocr.Reader(['en', 'id'], gpu=False)
    print("✅ Enhanced OCR Reader siap.")
except Exception as e:
    print(f"❌ CRITICAL: GAGAL MEMUAT OCR READER: {e}")
    print("Install EasyOCR dengan: pip install easyocr")

# ===== ENHANCED TEXT PROCESSING FUNCTIONS - PRODUCTION VERSION =====

def preprocess_image_for_ocr(image, method="default"):
    """
    Enhanced preprocessing dengan berbagai metode untuk OCR yang lebih baik
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == "default":
            return gray
        
        elif method == "enhanced_contrast":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return enhanced
        
        elif method == "threshold_otsu":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        
        elif method == "threshold_adaptive":
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            return thresh
        
        elif method == "denoised":
            denoised = cv2.fastNlMeansDenoising(gray)
            return denoised
        
        elif method == "sharpened":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            return sharpened
        
        elif method == "high_contrast":
            alpha = 2.5
            beta = -50
            high_contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            return high_contrast
        
        elif method == "morphology":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            return processed
        
        else:
            return gray
            
    except Exception as e:
        print(f"Error in preprocessing method {method}: {e}")
        return gray if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def enhanced_text_detection_system(image, target_text):
    """
    Enhanced text detection system dengan multiple approaches dan error handling
    """
    if ocr_reader is None:
        raise RuntimeError("OCR Reader tidak tersedia. Restart aplikasi atau install EasyOCR.")
    
    print(f"🔍 Enhanced Detection for: '{target_text}'")
    
    try:
        # Normalize target text
        target_normalized = re.sub(r'\s+', ' ', target_text.strip().lower())
        target_words = set(target_normalized.split())
        
        # Multiple preprocessing methods
        preprocessing_methods = [
            "default",
            "enhanced_contrast", 
            "threshold_otsu",
            "threshold_adaptive",
            "denoised",
            "sharpened",
            "high_contrast",
            "morphology"
        ]
        
        # Multiple scale factors
        scale_factors = [1.0, 1.5, 2.0, 0.8]
        
        all_detections = []
        best_match = None
        best_score = 0
        
        for scale in scale_factors:
            # Scale image if needed
            if scale != 1.0:
                h, w = image.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                if new_w > 0 and new_h > 0:
                    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                else:
                    continue
            else:
                scaled_image = image.copy()
            
            for method in preprocessing_methods:
                try:
                    print(f"   Testing: scale={scale:.1f}, method={method}")
                    
                    # Preprocess image
                    processed_image = preprocess_image_for_ocr(scaled_image, method)
                    
                    # OCR detection dengan parameter yang lebih sensitif
                    results = ocr_reader.readtext(
                        processed_image,
                        detail=1,
                        paragraph=False,
                        width_ths=0.7,
                        height_ths=0.7,
                        decoder='greedy'
                    )
                    
                    for (bbox, detected_text, confidence) in results:
                        if confidence < 0.05:  # Threshold sangat rendah
                            continue
                        
                        # Adjust bbox if scaled
                        if scale != 1.0:
                            bbox = [(x/scale, y/scale) for x, y in bbox]
                        
                        # Normalize detected text
                        detected_normalized = re.sub(r'\s+', ' ', detected_text.strip().lower())
                        detected_words = set(detected_normalized.split())
                        
                        # Multiple similarity scoring
                        similarity_scores = []
                        
                        # 1. Exact match (highest priority)
                        if target_normalized == detected_normalized:
                            similarity_scores.append(1.0)
                            print(f"      ✅ EXACT: '{detected_text}' | conf: {confidence:.3f}")
                        
                        # 2. Contains match
                        elif target_normalized in detected_normalized:
                            similarity_scores.append(0.95)
                            print(f"      🎯 CONTAINS (target in detected): '{detected_text}'")
                        elif detected_normalized in target_normalized:
                            similarity_scores.append(0.9)
                            print(f"      🎯 CONTAINS (detected in target): '{detected_text}'")
                        
                        # 3. Word overlap scoring
                        if len(target_words) > 0:
                            common_words = target_words.intersection(detected_words)
                            if len(common_words) > 0:
                                word_overlap = len(common_words) / len(target_words)
                                if word_overlap >= 0.5:  # At least 50% words match
                                    similarity_scores.append(word_overlap * 0.85)
                                    print(f"      📝 WORD OVERLAP ({word_overlap:.2f}): '{detected_text}' | common: {common_words}")
                        
                        # 4. Character overlap
                        target_chars = set(''.join(target_normalized.split()))
                        detected_chars = set(''.join(detected_normalized.split()))
                        if len(target_chars) > 0:
                            char_overlap = len(target_chars.intersection(detected_chars)) / len(target_chars)
                            if char_overlap >= 0.6:  # At least 60% chars match
                                similarity_scores.append(char_overlap * 0.7)
                                print(f"      🔤 CHAR OVERLAP ({char_overlap:.2f}): '{detected_text}'")
                        
                        # 5. Fuzzy matching (if available)
                        if fuzz:
                            fuzzy_ratio = fuzz.ratio(target_normalized, detected_normalized) / 100
                            if fuzzy_ratio >= 0.6:
                                similarity_scores.append(fuzzy_ratio * 0.8)
                                print(f"      🔄 FUZZY ({fuzzy_ratio:.2f}): '{detected_text}'")
                        
                        # 6. Partial string matching
                        if len(target_text) >= 3:
                            for i in range(len(target_text) - 2):
                                substr = target_text[i:i+3].lower()
                                if substr in detected_normalized:
                                    partial_score = 0.4 + (len(substr) / len(target_text)) * 0.3
                                    similarity_scores.append(partial_score)
                                    print(f"      🔍 PARTIAL MATCH: '{substr}' in '{detected_text}'")
                                    break
                        
                        # Calculate final score
                        if similarity_scores:
                            max_similarity = max(similarity_scores)
                            
                            # Boost score for high confidence
                            confidence_boost = min(confidence * 0.2, 0.15)
                            final_score = max_similarity + confidence_boost
                            
                            # Boost for exact length match
                            if abs(len(target_text) - len(detected_text)) <= 2:
                                final_score += 0.05
                            
                            detection_info = {
                                'bbox': bbox,
                                'text': detected_text,
                                'confidence': confidence,
                                'similarity': max_similarity,
                                'final_score': final_score,
                                'method': f"{method}_scale{scale:.1f}",
                                'scale': scale
                            }
                            
                            all_detections.append(detection_info)
                            
                            print(f"      📊 Score: {final_score:.3f} (sim: {max_similarity:.3f}, conf: {confidence:.3f})")
                            
                            if final_score > best_score:
                                best_score = final_score
                                best_match = detection_info
                
                except Exception as e:
                    print(f"      ❌ Error with {method} at scale {scale}: {e}")
                    continue
        
        # Jika tidak ada yang bagus, coba fuzzy search pada semua results
        if (not best_match or best_score < 0.3) and all_detections and fuzz and process:
            print("🔄 Trying fuzzy fallback search...")
            try:
                all_texts = [d['text'] for d in all_detections]
                fuzzy_result = process.extractOne(
                    target_text, 
                    all_texts, 
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=50
                )
                
                if fuzzy_result:
                    matched_text, fuzzy_score = fuzzy_result
                    matched_detection = next(d for d in all_detections if d['text'] == matched_text)
                    matched_detection['similarity'] = fuzzy_score / 100
                    matched_detection['final_score'] = (fuzzy_score / 100) * matched_detection['confidence']
                    
                    if matched_detection['final_score'] > best_score:
                        best_match = matched_detection
                        best_score = matched_detection['final_score']
                        print(f"   ✅ Fuzzy fallback: '{matched_text}' | score: {fuzzy_score}")
            
            except Exception as e:
                print(f"   ⚠️ Fuzzy fallback error: {e}")
        
        # Final attempt: word-by-word search
        if (not best_match or best_score < 0.25) and len(target_text.split()) > 1:
            print("🔄 Trying word-by-word detection...")
            target_words = target_text.split()
            
            for word in target_words:
                if len(word) >= 3:  # Skip short words
                    try:
                        word_result = enhanced_text_detection_system(image, word)
                        if word_result:
                            print(f"   ✅ Found word '{word}' from '{target_text}'")
                            return word_result
                    except:
                        continue
        
        # Print summary
        if best_match:
            print(f"✅ FOUND: '{best_match['text']}' | Score: {best_score:.3f} | Method: {best_match['method']}")
            return (best_match['bbox'], best_match['text'], best_match['confidence'], best_match['final_score'])
        else:
            print(f"❌ NOT FOUND: '{target_text}'")
            print("📝 All detected texts (top 15):")
            unique_texts = {}
            for d in all_detections:
                text = d['text']
                if text not in unique_texts or d['confidence'] > unique_texts[text]['confidence']:
                    unique_texts[text] = d
            
            sorted_detections = sorted(unique_texts.values(), key=lambda x: x['confidence'], reverse=True)[:15]
            for i, detection in enumerate(sorted_detections):
                print(f"   {i+1:2d}. '{detection['text']}' (conf: {detection['confidence']:.2f})")
            
            return None
    
    except Exception as e:
        print(f"❌ Critical error in text detection: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_text_region_mask(image, bbox, padding=5):
    """
    Extract mask for text region dengan improved padding dan error handling
    """
    try:
        # Validate input
        if bbox is None or len(bbox) < 3:
            print("Invalid bbox for mask creation")
            return None
        
        # Convert bbox to integer coordinates
        points = np.array(bbox, dtype=np.int32)
        
        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # Enhanced padding dengan morphological operations
        kernel_size = max(3, padding * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        return mask
        
    except Exception as e:
        print(f"Error creating text mask: {e}")
        return None

def intelligent_color_extraction(image, bbox, mask=None):
    """
    Intelligent color extraction untuk text dengan error handling
    """
    try:
        # Get bounding rectangle
        points = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        
        # Ensure coordinates are within image bounds
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            print("Invalid text region dimensions")
            return (0, 0, 0)
        
        # Extract text area
        text_area = image[y:y+h, x:x+w]
        
        if text_area.size == 0:
            return (0, 0, 0)
        
        # Convert to RGB for color analysis
        if len(text_area.shape) == 3:
            text_rgb = cv2.cvtColor(text_area, cv2.COLOR_BGR2RGB)
        else:
            text_rgb = cv2.cvtColor(text_area, cv2.COLOR_GRAY2RGB)
        
        # Multiple color detection methods
        color_candidates = []
        
        # Method 1: Edge-based detection
        try:
            gray = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY) if len(text_area.shape) == 3 else text_area
            edges = cv2.Canny(gray, 50, 150)
            
            if np.any(edges):
                edge_coords = np.where(edges > 0)
                if len(edge_coords[0]) > 0:
                    edge_pixels = text_rgb[edge_coords]
                    if len(edge_pixels) > 0:
                        edge_color = np.median(edge_pixels, axis=0)
                        color_candidates.append(('edge', edge_color))
        except Exception as e:
            print(f"Edge detection color extraction failed: {e}")
        
        # Method 2: Otsu thresholding
        try:
            gray = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY) if len(text_area.shape) == 3 else text_area
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract foreground and background
            fg_pixels = text_rgb[thresh == 0]  # Dark pixels (likely text)
            bg_pixels = text_rgb[thresh == 255]  # Light pixels (likely background)
            
            if len(fg_pixels) > 0 and len(bg_pixels) > 0:
                # Choose the one with higher contrast
                fg_color = np.median(fg_pixels, axis=0)
                bg_color = np.median(bg_pixels, axis=0)
                
                # Calculate contrast
                fg_luminance = 0.299 * fg_color[0] + 0.587 * fg_color[1] + 0.114 * fg_color[2]
                bg_luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                
                if abs(fg_luminance - bg_luminance) > 50:  # Good contrast
                    if len(fg_pixels) < len(bg_pixels):  # Assume text is minority
                        color_candidates.append(('otsu_fg', fg_color))
                    else:
                        color_candidates.append(('otsu_bg', bg_color))
        except Exception as e:
            print(f"Otsu color extraction failed: {e}")
        
        # Method 3: Dominant color using KMeans
        try:
            pixels = text_rgb.reshape((-1, 3))
            if len(pixels) > 10:  # Need enough pixels
                n_clusters = min(3, len(pixels))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Get cluster with median size (likely text)
                labels = kmeans.labels_
                unique, counts = np.unique(labels, return_counts=True)
                
                if len(counts) > 1:
                    sorted_indices = np.argsort(counts)
                    # Choose the second smallest cluster (often text)
                    chosen_idx = sorted_indices[min(1, len(sorted_indices)-1)]
                    dominant_color = kmeans.cluster_centers_[chosen_idx]
                    color_candidates.append(('kmeans', dominant_color))
        except Exception as e:
            print(f"KMeans color extraction failed: {e}")
        
        # Choose best color candidate
        if color_candidates:
            # Prioritize edge detection if available
            for method, color in color_candidates:
                if method == 'edge':
                    return tuple(np.round(color).astype(int))
            
            # Otherwise use the first available
            return tuple(np.round(color_candidates[0][1]).astype(int))
        
        # Fallback: use median color of entire region
        median_color = np.median(text_rgb.reshape((-1, 3)), axis=0)
        return tuple(np.round(median_color).astype(int))
        
    except Exception as e:
        print(f"Error in intelligent color extraction: {e}")
        return (0, 0, 0)

def estimate_font_characteristics(image, bbox, text):
    """
    Enhanced font characteristic estimation dengan error handling
    """
    try:
        points = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        
        # Ensure valid dimensions
        w, h = max(1, w), max(1, h)
        
        # Character-based size estimation
        char_count = len(text.replace(' ', ''))
        if char_count == 0:
            char_count = 1
        
        # Improved font size calculation
        avg_char_width = w / char_count
        estimated_font_size = int(h * 0.8)  # Text typically uses 80% of height
        
        # Adjust based on character width
        if avg_char_width > h * 0.8:  # Wide characters
            estimated_font_size = int(estimated_font_size * 0.9)
        elif avg_char_width < h * 0.4:  # Narrow characters
            estimated_font_size = int(estimated_font_size * 1.1)
        
        # Reasonable bounds
        estimated_font_size = max(8, min(estimated_font_size, 300))
        
        # Enhanced text analysis
        is_bold = False
        is_italic = False
        
        try:
            if (x >= 0 and y >= 0 and 
                x + w <= image.shape[1] and y + h <= image.shape[0] and
                w > 0 and h > 0):
                
                text_area = image[y:y+h, x:x+w]
                
                if text_area.size > 0:
                    # Convert to grayscale for analysis
                    gray_text = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY) if len(text_area.shape) == 3 else text_area
                    
                    # Bold detection (stroke width analysis)
                    edges = cv2.Canny(gray_text, 50, 150)
                    edge_density = np.sum(edges > 0) / (w * h) if w * h > 0 else 0
                    is_bold = edge_density > 0.08  # Adjusted threshold
                    
                    # Italic detection (slant analysis)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.contourArea(contour) > max(10, w * h * 0.05):
                            rect = cv2.minAreaRect(contour)
                            angle = abs(rect[2])
                            if 10 < angle < 80:  # Italic slant range
                                is_italic = True
                                break
        except Exception as e:
            print(f"Font style analysis failed: {e}")
        
        return {
            'size': estimated_font_size,
            'bold': is_bold,
            'italic': is_italic,
            'width': w,
            'height': h,
            'char_width': avg_char_width,
            'char_count': char_count
        }
        
    except Exception as e:
        print(f"Error estimating font characteristics: {e}")
        return {
            'size': 20,
            'bold': False,
            'italic': False,
            'width': 100,
            'height': 30,
            'char_width': 10,
            'char_count': 5
        }

def advanced_inpainting(image, mask):
    """
    Advanced background inpainting dengan multiple methods dan error handling
    """
    try:
        # Validate inputs
        if image is None or mask is None:
            print("Invalid input for inpainting")
            return image
        
        if image.shape[:2] != mask.shape[:2]:
            print("Image and mask dimensions don't match")
            return image
        
        # Method 1: OpenCV Telea (fast and good for small areas)
        result1 = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # Method 2: OpenCV NS (better for textures)
        result2 = cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)
        
        # Method 3: Custom patch-based inpainting for better results
        try:
            # Find mask region
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                
                # Expand region for context
                margin = max(5, min(w, h) // 2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)  
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                
                # Extract region with context
                region = image[y1:y2, x1:x2]
                region_mask = mask[y1:y2, x1:x2]
                
                if region.size > 0 and np.any(region_mask):
                    # Apply inpainting to region
                    inpainted_region = cv2.inpaint(region, region_mask, 7, cv2.INPAINT_NS)
                    
                    # Smooth the boundaries
                    kernel = np.ones((3, 3), np.float32) / 9
                    inpainted_region = cv2.filter2D(inpainted_region, -1, kernel)
                    
                    # Put back into result
                    result3 = image.copy()
                    result3[y1:y2, x1:x2] = inpainted_region
                else:
                    result3 = result2
            else:
                result3 = result2
                
        except Exception as e:
            print(f"Custom inpainting failed: {e}")
            result3 = result2
        
        # Return the most advanced method result
        return result3
        
    except Exception as e:
        print(f"Error in advanced inpainting: {e}")
        return image

def get_optimal_font(font_properties, text):
    """
    Get optimal font dengan error handling dan fallback yang lebih baik
    """
    # Font search paths untuk berbagai OS
    font_search_paths = [
        # Windows
        "C:/Windows/Fonts/",
        "C:/WINDOWS/Fonts/",
        # macOS
        "/System/Library/Fonts/",
        # Linux
        "/usr/share/fonts/truetype/dejavu/",
        "/usr/share/fonts/truetype/liberation/",
        "/usr/share/fonts/TTF/",
        # Current directory
        ""
    ]
    
    # Font candidates dengan style variations
    font_candidates = [
        # Arial family - most common
        ("arial.ttf", "Arial", False, False),
        ("arialbd.ttf", "Arial Bold", True, False),
        ("ariali.ttf", "Arial Italic", False, True),
        ("arialbi.ttf", "Arial Bold Italic", True, True),
        
        # DejaVu (Linux default)
        ("DejaVuSans.ttf", "DejaVu Sans", False, False),
        ("DejaVuSans-Bold.ttf", "DejaVu Sans Bold", True, False),
        
        # Liberation (Linux)
        ("LiberationSans-Regular.ttf", "Liberation Sans", False, False),
        ("LiberationSans-Bold.ttf", "Liberation Sans Bold", True, False),
        
        # Helvetica (macOS)
        ("Helvetica.ttf", "Helvetica", False, False),
        ("Helvetica-Bold.ttf", "Helvetica Bold", True, False),
        
        # Other common fonts
        ("calibri.ttf", "Calibri", False, False),
        ("calibrib.ttf", "Calibri Bold", True, False),
        ("times.ttf", "Times New Roman", False, False),
        ("timesbd.ttf", "Times New Roman Bold", True, False),
        ("verdana.ttf", "Verdana", False, False),
        ("tahoma.ttf", "Tahoma", False, False),
    ]
    
    target_bold = font_properties.get('bold', False)
    target_italic = font_properties.get('italic', False) 
    target_size = max(8, min(font_properties.get('size', 20), 200))  # Reasonable bounds
    
    best_font = None
    best_score = -1
    
    for font_file, font_name, is_bold, is_italic in font_candidates:
        # Score based on style matching
        style_score = 0
        if is_bold == target_bold:
            style_score += 2
        if is_italic == target_italic:
            style_score += 1
        
        # Skip if style doesn't match well
        if style_score == 0:
            continue
        
        # Try to load font from various paths
        for base_path in font_search_paths:
            font_path = os.path.join(base_path, font_file)
            
            if os.path.exists(font_path):
                try:
                    # Test font loading
                    test_font = ImageFont.truetype(font_path, target_size)
                    
                    # Quick test - ensure font can render text
                    test_img = Image.new('RGB', (100, 50), 'white')
                    test_draw = ImageDraw.Draw(test_img)
                    
                    # Try to get text dimensions
                    test_text = text[:10] if len(text) > 10 else text  # Test with first 10 chars
                    bbox = test_draw.textbbox((0, 0), test_text, font=test_font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    if text_width > 0 and text_height > 0:  # Valid font
                        if style_score > best_score:
                            best_font = test_font
                            best_score = style_score
                            print(f"✅ Selected font: {font_name} at {target_size}px (score: {style_score})")
                        break
                        
                except Exception as e:
                    print(f"   Failed to load {font_path}: {e}")
                    continue
    
    # Ultimate fallback options
    if best_font is None:
        try:
            # Try system default font
            best_font = ImageFont.load_default()
            print("⚠️ Using system default font")
        except Exception as e:
            print(f"❌ Could not load any font: {e}")
            best_font = None
    
    return best_font

def intelligent_text_rendering(image_pil, text, bbox, text_color, font_properties):
    """
    Intelligent text rendering dengan positioning yang lebih akurat dan error handling
    """
    try:
        print(f"🎨 Rendering text: '{text}' with color {text_color}")
        
        # Get optimal font
        font = get_optimal_font(font_properties, text)
        if font is None:
            print("❌ Critical: Cannot load any font")
            return image_pil
        
        # Calculate text region
        points = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        
        # Ensure valid region
        if w <= 0 or h <= 0:
            print("Invalid text region for rendering")
            return image_pil
        
        # Create drawing context
        draw = ImageDraw.Draw(image_pil)
        
        # Dynamic font size adjustment
        original_size = font_properties.get('size', 20)
        current_size = original_size
        
        # Test current font size and adjust if needed
        adjustment_attempts = 0
        while current_size >= 8 and adjustment_attempts < 10:  # Prevent infinite loop
            try:
                # Try to get font at current size
                if hasattr(font, 'size') and getattr(font, 'size', None) != current_size:
                    font = get_optimal_font({**font_properties, 'size': current_size}, text)
                    if font is None:
                        font = ImageFont.load_default()
                
                # Measure text
                bbox_test = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_test[2] - bbox_test[0]
                text_height = bbox_test[3] - bbox_test[1]
                
                # Check if text fits with some margin
                if text_width <= w * 0.95 and text_height <= h * 0.95:
                    break
                else:
                    # Reduce size by 10%
                    current_size = max(8, int(current_size * 0.9))
                    adjustment_attempts += 1
                    
            except Exception as e:
                print(f"Font adjustment error: {e}")
                current_size = max(8, int(current_size * 0.9))
                adjustment_attempts += 1
        
        # Final text positioning - center in bounding box
        try:
            final_bbox = draw.textbbox((0, 0), text, font=font)
            final_width = final_bbox[2] - final_bbox[0]
            final_height = final_bbox[3] - final_bbox[1]
        except Exception as e:
            print(f"Text measurement error: {e}")
            final_width, final_height = w, h  # Fallback
        
        # Calculate centered position
        text_x = x + max(0, (w - final_width) // 2)
        text_y = y + max(0, (h - final_height) // 2)
        
        # Ensure text doesn't go outside image boundaries
        text_x = max(0, min(text_x, image_pil.width - final_width))
        text_y = max(0, min(text_y, image_pil.height - final_height))
        
        # Render text dengan error handling
        try:
            draw.text((text_x, text_y), text, fill=text_color, font=font)
            print(f"✅ Text rendered at ({text_x}, {text_y}) with size {current_size}")
        except Exception as e:
            print(f"Text rendering error: {e}")
            # Fallback dengan default font
            try:
                default_font = ImageFont.load_default()
                draw.text((text_x, text_y), text, fill=text_color, font=default_font)
                print("⚠️ Used default font for rendering")
            except Exception as e2:
                print(f"❌ Even default font rendering failed: {e2}")
        
        return image_pil
        
    except Exception as e:
        print(f"❌ Error in intelligent text rendering: {e}")
        import traceback
        traceback.print_exc()
        return image_pil

def enhanced_text_swap_pipeline(image_pil, original_text, new_text):
    """
    Complete enhanced text swap pipeline - PRODUCTION VERSION dengan comprehensive error handling
    """
    try:
        print("🚀 Starting Enhanced Text Swap Pipeline (Production Version)")
        print(f"📝 Target: '{original_text}' → '{new_text}'")
        
        # Validate OCR availability
        if ocr_reader is None:
            raise RuntimeError("OCR Reader tidak tersedia. Silakan restart server atau install EasyOCR.")
        
        # Convert PIL to OpenCV format
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        print("🔍 Phase 1: Enhanced Text Detection")
        detection_result = enhanced_text_detection_system(image_bgr, original_text)
        
        if detection_result is None:
            # Try with individual words as fallback
            words = original_text.split()
            if len(words) > 1:
                print("🔄 Fallback: Trying individual words...")
                for word in words:
                    if len(word) >= 3:  # Skip very short words
                        print(f"   🔍 Trying word: '{word}'")
                        try:
                            detection_result = enhanced_text_detection_system(image_bgr, word)
                            if detection_result:
                                print(f"   ✅ Success with word: '{word}'")
                                break
                        except Exception as e:
                            print(f"   ❌ Word detection failed: {e}")
                            continue
        
        if detection_result is None:
            # Provide helpful error message with detected texts
            available_texts = []
            try:
                results = ocr_reader.readtext(image_bgr, detail=1)
                available_texts = [
                    text for (_, text, conf) in results 
                    if conf > 0.1 and len(text.strip()) > 0
                ]
            except Exception as e:
                print(f"Error getting available texts: {e}")
            
            error_message = f"❌ Text '{original_text}' tidak ditemukan dalam gambar."
            
            if available_texts:
                # Sort by length and remove duplicates
                available_texts.sort(key=len, reverse=True)
                unique_texts = list(dict.fromkeys(available_texts))[:10]
                error_message += f" Text yang terdeteksi: {', '.join(unique_texts)}"
            else:
                error_message += " Pastikan gambar memiliki kualitas yang baik dan text terlihat jelas."
            
            raise ValueError(error_message)
        
        bbox, detected_text, confidence, score = detection_result
        print(f"✅ Text detected: '{detected_text}' (confidence: {confidence:.3f}, score: {score:.3f})")
        
        print("🎯 Phase 2: Region Analysis & Mask Creation")
        text_mask = extract_text_region_mask(image_bgr, bbox, padding=3)
        
        if text_mask is None:
            raise ValueError("❌ Failed to create text mask from detected region")
        
        print("🎨 Phase 3: Color & Font Analysis")
        text_color = intelligent_color_extraction(image_bgr, bbox, text_mask)
        font_properties = estimate_font_characteristics(image_bgr, bbox, detected_text)
        
        print(f"📊 Analysis Results:")
        print(f"   Color: {text_color}")
        print(f"   Font: {font_properties}")
        
        print("🔧 Phase 4: Advanced Background Inpainting")
        inpainted_bgr = advanced_inpainting(image_bgr, text_mask)
        
        # Convert back to RGB PIL format
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(inpainted_rgb)
        
        print("✍️ Phase 5: Intelligent Text Rendering")
        final_image = intelligent_text_rendering(
            result_image, 
            new_text, 
            bbox, 
            text_color, 
            font_properties
        )
        
        print("✅ Enhanced Text Swap Pipeline Complete!")
        return final_image, detected_text, confidence
        
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        raise e

def process_image(file_storage):
    """Process uploaded image dengan error handling"""
    try:
        in_memory_file = np.frombuffer(file_storage.read(), np.uint8)
        return cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

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

# --- 7. ENHANCED Text Swap Process - PRODUCTION VERSION ---
@app.route('/text-swap-process', methods=['POST'])
@login_required
def text_swap_process():
    """Enhanced text swap process dengan comprehensive error handling"""
    
    # Validate OCR Reader availability
    if ocr_reader is None:
        return jsonify({
            'error': 'OCR Reader tidak berhasil dimuat di server. Silakan restart aplikasi atau install EasyOCR dengan: pip install easyocr',
            'suggestions': [
                '🔧 Restart server aplikasi',
                '📦 Install EasyOCR: pip install easyocr',
                '🐳 Pastikan dependencies terinstall dengan benar'
            ]
        }), 500
    
    # Validate request
    if 'image' not in request.files:
        return jsonify({
            'error': 'Gambar dibutuhkan untuk proses text swap',
            'suggestions': ['📷 Upload gambar yang mengandung text yang ingin diganti']
        }), 400
    
    image_file = request.files['image']
    original_text = request.form.get('original_text', '').strip()
    new_text = request.form.get('new_text', '').strip()
    
    # Validate inputs
    if not original_text or not new_text:
        return jsonify({
            'error': 'Text asli dan text pengganti dibutuhkan',
            'suggestions': [
                '📝 Masukkan text yang ingin dicari di gambar',
                '✏️ Masukkan text pengganti',
                '💡 Contoh: "SPIDER-MAN" → "LABA-LABA"'
            ]
        }), 400
    
    if len(original_text) < 2:
        return jsonify({
            'error': 'Text asli terlalu pendek (minimal 2 karakter)',
            'suggestions': ['📏 Gunakan text yang lebih panjang untuk detection yang akurat']
        }), 400
    
    try:
        # Reset file pointer and validate image
        image_file.seek(0)
        
        try:
            image = Image.open(image_file).convert('RGB')
        except Exception as e:
            return jsonify({
                'error': f'Error membaca gambar: {str(e)}',
                'suggestions': [
                    '🖼️ Pastikan file adalah gambar valid (JPG, PNG, etc.)',
                    '📱 Coba compress gambar jika ukurannya terlalu besar',
                    '🔄 Coba upload ulang dengan gambar yang berbeda'
                ]
            }), 400
        
        # Validate image size
        if image.width < 100 or image.height < 100:
            return jsonify({
                'error': 'Gambar terlalu kecil (minimal 100x100 pixels)',
                'suggestions': [
                    '📏 Gunakan gambar dengan resolusi minimal 100x100 pixels',
                    '🔍 Pastikan text di gambar terlihat jelas'
                ]
            }), 400
        
        # Resize very large images for performance
        if image.width > 4000 or image.height > 4000:
            ratio = min(4000/image.width, 4000/image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, RESAMPLE_FILTER)
            print(f"📏 Image resized to {new_size} for processing")
        
        print(f"🚀 Processing text swap: '{original_text}' → '{new_text}'")
        
        # Process text swap using enhanced pipeline
        result_image, detected_text, confidence = enhanced_text_swap_pipeline(
            image, original_text, new_text
        )
        
        # Save result to server
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        unique_id = str(uuid.uuid4())
        result_filename = f"{unique_id}_text_swap_enhanced.jpg"
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
        print(f"❌ Error dalam text swap: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Enhanced error categorization dan suggestions
        suggestions = []
        debug_info = ""
        
        if "tidak ditemukan" in error_msg.lower() or "not found" in error_msg.lower():
            suggestions = [
                "🔍 Pastikan text yang dicari benar-benar ada dan terlihat jelas di gambar",
                "🔤 Coba variasi penulisan: 'Jepang', 'JEPANG', atau 'jepang'",
                "✂️ Coba gunakan sebagian dari text (misal: 'Jepang' dari 'Wisata Jepang')",
                "🖼️ Pastikan gambar memiliki kualitas HD dan text tidak terpotong",
                "🎯 Gunakan text yang unik/spesifik, hindari kata umum seperti 'the', 'dan'"
            ]
            
            # Extract available texts for debugging
            if "Text yang terdeteksi:" in error_msg:
                debug_info = "💡 Lihat text yang berhasil terdeteksi dalam pesan error di atas"
            
        elif "mask" in error_msg.lower():
            suggestions = [
                "🎯 Text berhasil ditemukan tapi gagal membuat area replacement",
                "🔄 Coba dengan gambar yang berbeda atau crop area text saja", 
                "🔍 Pastikan text tidak terlalu kecil, terdistorsi, atau overlap dengan objek lain"
            ]
        
        elif "OCR" in error_msg.upper() or "Reader" in error_msg:
            suggestions = [
                "🔧 Sistem OCR bermasalah, silakan restart aplikasi",
                "📦 Pastikan EasyOCR terinstall dengan benar: pip install easyocr",
                "🖼️ Coba dengan gambar format JPG atau PNG yang valid"
            ]
        
        elif "font" in error_msg.lower():
            suggestions = [
                "🔤 Masalah dengan font rendering",
                "🔄 Text berhasil dideteksi tapi gagal render text baru",
                "💻 Pastikan sistem memiliki font yang dibutuhkan"
            ]
        
        else:
            suggestions = [
                "🔧 Terjadi kesalahan teknis dalam pemrosesan gambar",
                "📱 Coba refresh halaman dan upload ulang",
                "🖼️ Pastikan format gambar adalah JPG/PNG yang valid",
                "💾 Pastikan ukuran file gambar tidak terlalu besar (maksimal 10MB)"
            ]
        
        error_response = {
            'error': error_msg,
            'suggestions': suggestions,
            'debug_tip': "💡 Untuk hasil terbaik: gunakan gambar HD dengan text yang kontras dan jelas terlihat"
        }
        
        if debug_info:
            error_response['debug_info'] = debug_info
        
        return jsonify(error_response), 500

# --- 8. Rute API untuk Face Swap (Tidak berubah) ---
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
        
        if source_img is None or target_img is None:
            return jsonify({'error': 'Gagal memproses gambar yang diupload'}), 400
        
        source_faces = face_analyzer.get(source_img)
        target_faces = face_analyzer.get(target_img)

        if not source_faces or not target_faces:
            return jsonify({'error': 'Wajah tidak terdeteksi di salah satu gambar'}), 400
        
        source_face = source_faces[0]
        target_face = max(target_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        result_img = swapper.get(target_img, target_face, source_face, paste_back=True)

        # === JALANKAN PROSES UPSCALING VIA REPLICATE API JIKA DIMINTA ===
        if enhance_quality:
            try:
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
                    print("✅ Upscaling berhasil")
                else:
                    print("⚠️ Gagal mengunduh gambar hasil dari Replicate.")
            except Exception as e:
                print(f"⚠️ Upscaling gagal: {e}")
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

# --- 10. Testing Function untuk Development ---
def test_text_detection_manual(image_path, target_text):
    """
    Fungsi untuk testing manual dari command line - untuk development
    """
    if not ocr_reader:
        print("❌ OCR Reader tidak tersedia")
        return
    
    print("="*60)
    print(f"TESTING ENHANCED TEXT DETECTION")
    print(f"Image: {image_path}")
    print(f"Target: '{target_text}'")
    print("="*60)
    
    # Load image
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"❌ Cannot load image: {image_path}")
            return
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Test detection
    try:
        result = enhanced_text_detection_system(image_bgr, target_text)
        
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
            
    except Exception as e:
        print(f"❌ Detection error: {e}")
    
    print("="*60)

# --- 11. Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint untuk monitoring"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'face_analyzer': face_analyzer is not None,
                'face_swapper': swapper is not None,
                'ocr_reader': ocr_reader is not None,
                'database': True,  # Asumsi database selalu ready
                'fuzzywuzzy': fuzz is not None
            }
        }
        
        # Test database connection
        try:
            db.session.execute('SELECT 1')
            status['services']['database'] = True
        except Exception as e:
            status['services']['database'] = False
            status['database_error'] = str(e)
        
        # Determine overall health
        critical_services = ['database', 'ocr_reader']
        healthy = all(status['services'].get(service, False) for service in critical_services)
        
        if not healthy:
            status['status'] = 'unhealthy'
            return jsonify(status), 503
        
        return jsonify(status), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# --- 12. Menjalankan Aplikasi & Membuat Database ---
if __name__ == '__main__':
    # Uncomment untuk testing manual text detection
    # test_text_detection_manual("path/to/test/image.jpg", "TARGET_TEXT")
    
    # Jalankan app
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
    
    print("🚀 Starting AI Swapper Enhanced Server...")
    print("📝 Text Swapper: PRODUCTION VERSION with Enhanced Detection")
    print("🔥 Face Swapper: Ready with Replicate API integration")
    
    app.run(host='0.0.0.0', port=5000, debug=True)