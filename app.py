# =============================================================================
# Flask Web Application
# Real-Time OCR System for Video-Based Text Recognition
# in Traffic and Surveillance
# =============================================================================
#
# OCR Engine: PaddleOCR (PP-OCR detector + recognizer)
#   - PP-OCR: Deep learning text detector + recognizer in one pipeline
#   - Works on CPU, no GPU required
#   - Automatically detects text from running video — no manual selection needed
#
# Technologies: Python, Flask, OpenCV, PaddleOCR, NumPy
# =============================================================================

import os
os.environ.setdefault("FLAGS_use_pir", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from flask import Flask, render_template, request, jsonify, send_from_directory  # type: ignore
import cv2  # type: ignore
import numpy as np  # type: ignore
from paddleocr import PaddleOCR  # type: ignore
import re
import time
import base64
import uuid

app = Flask(__name__)

# =============================================================================
# PaddleOCR Initialization (loaded ONCE at startup)
# =============================================================================
# PaddleOCR uses PP-OCR for text detection + recognition.
# It automatically finds all text regions in an image and reads them.
# First run downloads model files (~100MB) — one-time download.
# =============================================================================

print("=" * 60)
print("  Loading PaddleOCR Models (PP-OCR)...")
print("  First run downloads models — one-time ~100MB")
print("=" * 60)

# Initialize PaddleOCR
# - lang='en' = English language
# - device='cpu' = CPU mode (set to 'gpu' if you have NVIDIA GPU + CUDA)
ocr_engine = PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    device='cpu',
    enable_mkldnn=False,
    enable_cinn=False,
)

print("[INFO] PaddleOCR loaded successfully!")

# ─── Upload folder ────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# ─── OCR Quality Constants ────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.15
OCR_DET_LIMIT_SIDE = 1280
OCR_DET_THRESH = 0.20
OCR_DET_BOX_THRESH = 0.30
OCR_DET_UNCLIP = 1.6
OCR_REC_SCORE_THRESH = 0.20
OCR_UPSCALE_MAX = 1280

PLATE_ONLY_MODE = True
PLATE_MAX_CANDIDATES = 8
PLATE_ASPECT_MIN = 2.0
PLATE_ASPECT_MAX = 6.5
PLATE_MIN_AREA_RATIO = 0.0008
PLATE_MAX_AREA_RATIO = 0.08
PLATE_MIN_W = 60
PLATE_MIN_H = 16
PLATE_BOX_EXPAND = 0.08

NOISE_PATTERN = re.compile(r'^[\W_]+$')
PLATE_TEXT_PATTERN = re.compile(r'^[A-Z0-9\- ]+$', re.IGNORECASE)


def _is_valid_text(text):
    """Check if detected text is real (not noise)."""
    text = text.strip()
    if len(text) < 1:
        return False
    if NOISE_PATTERN.match(text):
        return False
    if not re.search(r'[a-zA-Z0-9]', text):
        return False
    return True


def _is_plate_text(text):
    text = text.strip()
    if len(text) < 4:
        return False
    if not re.search(r'[0-9]', text):
        return False
    if not PLATE_TEXT_PATTERN.match(text):
        return False
    return True


def _prepare_frame_for_ocr(frame):
    h, w = frame.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    if max_side < OCR_UPSCALE_MAX:
        scale = OCR_UPSCALE_MAX / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return frame, scale


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter_area / float(area_a + area_b - inter_area + 1e-6)


def _merge_overlaps(boxes, iou_thresh=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    merged = []
    for box in boxes:
        merged_box = None
        for idx, existing in enumerate(merged):
            if _iou(box, existing) >= iou_thresh:
                merged_box = (
                    min(box[0], existing[0]),
                    min(box[1], existing[1]),
                    max(box[2], existing[2]),
                    max(box[3], existing[3]),
                )
                merged[idx] = merged_box
                break
        if merged_box is None:
            merged.append(box)
    return merged


def _expand_box(box, frame_w, frame_h, expand_ratio):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * expand_ratio)
    pad_h = int(bh * expand_ratio)
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(frame_w, x2 + pad_w)
    ny2 = min(frame_h, y2 + pad_h)
    return nx1, ny1, nx2, ny2


def _find_plate_candidates(frame):
    h, w = frame.shape[:2]
    min_area = int(h * w * PLATE_MIN_AREA_RATIO)
    max_area = int(h * w * PLATE_MAX_AREA_RATIO)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        if bw < PLATE_MIN_W or bh < PLATE_MIN_H:
            continue
        aspect = bw / float(bh + 1e-6)
        if aspect < PLATE_ASPECT_MIN or aspect > PLATE_ASPECT_MAX:
            continue
        candidates.append((x, y, x + bw, y + bh))

    candidates = _merge_overlaps(candidates)
    candidates = sorted(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return candidates[:PLATE_MAX_CANDIDATES]


# =============================================================================
# PaddleOCR Runner — Automatic Text Detection + Recognition
# =============================================================================

def run_paddleocr(image):
    """
    Run PaddleOCR on an image — automatically detects and reads ALL text.

    PaddleOCR does BOTH detection + recognition in one pipeline.

    Returns list of dicts:
      [{'bbox': [x1,y1,x2,y2], 'text': str, 'confidence': float}]
    """
    try:
        ocr_frame, scale = _prepare_frame_for_ocr(image)
        results = ocr_engine.predict(
            ocr_frame,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=OCR_DET_LIMIT_SIDE,
            text_det_limit_type="max",
            text_det_thresh=OCR_DET_THRESH,
            text_det_box_thresh=OCR_DET_BOX_THRESH,
            text_det_unclip_ratio=OCR_DET_UNCLIP,
            text_rec_score_thresh=OCR_REC_SCORE_THRESH,
        )
        detections = []

        if not results:
            return detections

        for res in results:
            if not isinstance(res, dict):
                continue

            polys = res.get('rec_polys') or []
            texts = res.get('rec_texts') or []
            scores = res.get('rec_scores') or []

            for poly, text_item, conf in zip(polys, texts, scores):
                text = text_item[0] if isinstance(text_item, (list, tuple)) else text_item
                text = str(text).strip()
                conf = float(conf)

                if conf < MIN_CONFIDENCE or not _is_valid_text(text):
                    continue

                xs = [int(p[0]) for p in poly]
                ys = [int(p[1]) for p in poly]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                if scale != 1.0:
                    x1 = int(x1 / scale)
                    x2 = int(x2 / scale)
                    y1 = int(y1 / scale)
                    y2 = int(y2 / scale)

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'text': text,
                    'confidence': round(float(conf), 3)
                })

        return detections

    except Exception as e:
        print(f"[WARNING] PaddleOCR error: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_plate_ocr(frame):
    h, w = frame.shape[:2]
    candidates = _find_plate_candidates(frame)
    detections = []

    for box in candidates:
        x1, y1, x2, y2 = _expand_box(box, w, h, PLATE_BOX_EXPAND)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_dets = run_paddleocr(crop)
        for det in crop_dets:
            if not _is_plate_text(det['text']):
                continue

            bx1, by1, bx2, by2 = det['bbox']
            detections.append({
                'bbox': [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                'text': det['text'],
                'confidence': det['confidence'],
            })

    return detections


def run_ocr_on_region(image):
    """Run PaddleOCR on a cropped region (manual selection mode)."""
    detections = run_paddleocr(image)
    if detections:
        texts = [d['text'] for d in detections]
        return " ".join(texts)
    return "No text detected"


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload."""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file"}), 400

        file = request.files['video']
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        filename_str = str(file.filename)
        ext = filename_str.rsplit('.', 1)[1].lower() if '.' in filename_str else 'mp4'
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f".{ext} not supported"}), 400

        safe_name = f"video_{uuid.uuid4().hex[:8]}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        file.save(filepath)

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({"error": "File save failed"}), 500

        print(f"[OK] Uploaded: {file.filename} -> {safe_name}")
        return jsonify({"status": "uploaded", "filename": safe_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/uploads/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/ocr_region', methods=['POST'])
def ocr_region():
    """Manual selection: crop region, run OCR, return zoomed + text."""
    if 'frame' not in request.files:
        return jsonify({"error": "No frame data"}), 400

    file = request.files['frame']
    coords = request.form

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        x1 = int(float(coords.get('x1', '0')))
        y1 = int(float(coords.get('y1', '0')))
        x2 = int(float(coords.get('x2', '0')))
        y2 = int(float(coords.get('y2', '0')))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid coordinates"}), 400

    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return jsonify({"error": "Invalid region"}), 400

    cropped = frame[y1:y2, x1:x2]
    text = run_ocr_on_region(cropped)

    # Create zoomed version
    crop_h, crop_w = cropped.shape[:2]
    zoom_w = 500
    zoom_scale = zoom_w / crop_w if crop_w > 0 else 1
    zoom_h = max(int(crop_h * zoom_scale), 50)
    zoomed = cv2.resize(cropped, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

    _, buffer = cv2.imencode('.jpg', zoomed, [cv2.IMWRITE_JPEG_QUALITY, 90])
    zoomed_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "text": text if text else "No text detected",
        "zoomed_image": f"data:image/jpeg;base64,{zoomed_b64}"
    })


# =============================================================================
# AUTO-DETECT ENDPOINT — Automatic Text Detection from Running Video
# =============================================================================
# This is the KEY feature: the browser sends a frame every ~1 second,
# PaddleOCR finds ALL text automatically, draws boxes, shows results in sidebar,
# and auto-zooms the best detection.
# =============================================================================

@app.route('/ocr_auto_detect', methods=['POST'])
def ocr_auto_detect():
    """
    Automatically detect and recognize ALL text in a video frame.
    No manual rectangle needed — PaddleOCR does it all.
    """
    if 'frame' not in request.files:
        return jsonify({"error": "No frame data"}), 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    start_time = time.time()

    # Run OCR on detected number plate candidates (plate-only mode)
    if PLATE_ONLY_MODE:
        detections = run_plate_ocr(frame)
    else:
        detections = run_paddleocr(frame)

    process_time = (time.time() - start_time) * 1000  # ms

    # Find best detection (highest confidence) for auto-zoom
    zoomed_b64 = None
    zoomed_text = None
    best_det = None

    for det in detections:
        if best_det is None or det['confidence'] > best_det['confidence']:
            best_det = det

    # Auto-zoom the best detected text region
    if best_det:
        bx1, by1, bx2, by2 = best_det['bbox']
        h, w = frame.shape[:2]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)

        if bx2 > bx1 and by2 > by1:
            cropped = frame[by1:by2, bx1:bx2]
            crop_h, crop_w = cropped.shape[:2]
            zoom_w = 400
            zoom_scale = zoom_w / crop_w if crop_w > 0 else 1
            zoom_h = max(int(crop_h * zoom_scale), 50)
            if zoom_w > 0 and zoom_h > 0:
                zoomed = cv2.resize(cropped, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
                _, buffer = cv2.imencode('.jpg', zoomed, [cv2.IMWRITE_JPEG_QUALITY, 90])
                zoomed_b64 = base64.b64encode(buffer).decode('utf-8')
                zoomed_text = best_det['text']

    return jsonify({
        "detections": detections,
        "process_time_ms": round(process_time, 1),
        "zoomed_image": f"data:image/jpeg;base64,{zoomed_b64}" if zoomed_b64 else None,
        "zoomed_text": zoomed_text
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 500MB."}), 413


# =============================================================================
# Run the Flask Server
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  Real-Time OCR System for Video-Based Text Recognition")
    print("  in Traffic and Surveillance")
    print("")
    print("  Engine: PaddleOCR (PP-OCR detection + recognition)")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
