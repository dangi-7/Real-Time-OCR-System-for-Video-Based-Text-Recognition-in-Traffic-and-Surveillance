"""
Real-Time OCR System for Video-Based Text Recognition
Standalone OpenCV app using PaddleOCR.

Controls:
  - Press 'a' to toggle auto-detect mode
  - Click and drag to select a region (manual mode)
  - Press 'c' to clear manual selection
  - Press 'q' to quit
"""

import os
os.environ.setdefault("FLAGS_use_pir", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2  # type: ignore
import re
import sys
import time

from paddleocr import PaddleOCR  # type: ignore

MIN_CONFIDENCE = 0.15
MIN_TEXT_LENGTH = 1
FRAME_SKIP = 8
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

USE_WEBCAM = False
VIDEO_FILE = "sample_video.mp4"


def _is_valid_text(text: str) -> bool:
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False
    if NOISE_PATTERN.match(text):
        return False
    if not re.search(r'[a-zA-Z0-9]', text):
        return False
    return True


def _is_plate_text(text: str) -> bool:
    text = text.strip()
    if len(text) < 4:
        return False
    if not re.search(r'[0-9]', text):
        return False
    if not PLATE_TEXT_PATTERN.match(text):
        return False
    return True


def open_video_source():
    if USE_WEBCAM:
        print("[INFO] Opening webcam (camera index 0)...")
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.exists(VIDEO_FILE):
            print(f"[ERROR] Video file not found: {VIDEO_FILE}")
            print("[ERROR] Place your video file in the project folder, or change VIDEO_FILE variable.")
            sys.exit(1)
        print(f"[INFO] Opening video file: {VIDEO_FILE}")
        cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        print("[ERROR] Could not open video source. Check your camera or file path.")
        sys.exit(1)

    return cap


def _get_confidence_color(confidence: float) -> tuple:
    if confidence >= 0.70:
        return (0, 255, 0)
    if confidence >= 0.40:
        return (0, 255, 255)
    return (0, 100, 255)


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


def run_paddleocr(image, ocr_engine):
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
    except Exception as exc:
        print(f"[WARNING] PaddleOCR error: {exc}")
        return []

    detections = []
    if not results:
        return detections

    for res in results:
        if not isinstance(res, dict):
            continue

        polys = res.get("rec_polys") or []
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []

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
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "confidence": conf,
            })

    return detections


def run_plate_ocr(frame, ocr_engine):
    h, w = frame.shape[:2]
    candidates = _find_plate_candidates(frame)
    detections = []

    for box in candidates:
        x1, y1, x2, y2 = _expand_box(box, w, h, PLATE_BOX_EXPAND)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_dets = run_paddleocr(crop, ocr_engine)
        for det in crop_dets:
            if not _is_plate_text(det["text"]):
                continue
            bx1, by1, bx2, by2 = det["bbox"]
            detections.append({
                "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                "text": det["text"],
                "confidence": det["confidence"],
            })

    return detections


mouse = {
    "drawing": False,
    "start_x": 0,
    "start_y": 0,
    "end_x": 0,
    "end_y": 0,
    "rect_ready": False,
}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse["drawing"] = True
        mouse["start_x"] = x
        mouse["start_y"] = y
        mouse["end_x"] = x
        mouse["end_y"] = y
        mouse["rect_ready"] = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse["drawing"]:
        mouse["end_x"] = x
        mouse["end_y"] = y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse["drawing"] = False
        mouse["end_x"] = x
        mouse["end_y"] = y

        dx = abs(mouse["end_x"] - mouse["start_x"])
        dy = abs(mouse["end_y"] - mouse["start_y"])
        if dx > 10 and dy > 10:
            mouse["rect_ready"] = True


def main():
    print("=" * 60)
    print("  Loading PaddleOCR (PP-OCR)...")
    print("  First run downloads models - one-time ~100MB")
    print("=" * 60)

    ocr_engine = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        device='cpu',
        enable_mkldnn=False,
        enable_cinn=False,
    )

    cap = open_video_source()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Video FPS: {fps:.0f}")

    cv2.namedWindow("Real-Time OCR Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time OCR Video", 960, 540)
    cv2.setMouseCallback("Real-Time OCR Video", mouse_callback)

    print("=" * 60)
    print("  REAL-TIME OCR SYSTEM - RUNNING")
    print("  Engine: PaddleOCR (PP-OCR detection + recognition)")
    print("")
    print("  Press 'a' to toggle auto-detect mode")
    print("  Draw a rectangle with your mouse for manual OCR")
    print("  Press 'c' to clear, 'q' to quit")
    print("=" * 60)

    auto_detect_mode = False
    auto_detections = []
    last_ocr_text = ""
    last_rect = []
    frame_count = 0
    fps_display = 0.0
    fps_timer = time.time()
    fps_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if not USE_WEBCAM:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[INFO] Camera disconnected.")
            break

        frame_count += 1

        display_w = 960
        h, w = frame.shape[:2]
        scale = display_w / w
        display_h = int(h * scale)
        display = cv2.resize(frame, (display_w, display_h))

        fps_frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_timer = time.time()

        if auto_detect_mode and frame_count % FRAME_SKIP == 0:
            if PLATE_ONLY_MODE:
                auto_detections = run_plate_ocr(display, ocr_engine)
            else:
                auto_detections = run_paddleocr(display, ocr_engine)

        if auto_detect_mode:
            for det in auto_detections:
                x1, y1, x2, y2 = det["bbox"]
                text = det["text"]
                conf = det["confidence"]
                color = _get_confidence_color(conf)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label = f"{text} ({int(conf * 100)}%)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                tw, th = cv2.getTextSize(label, font, font_scale, thickness)[0]
                label_y = y1 - 8
                if label_y < 20:
                    label_y = y2 + th + 8

                cv2.rectangle(
                    display,
                    (x1, label_y - th - 4),
                    (x1 + tw + 8, label_y + 4),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(display, label, (x1 + 4, label_y - 2), font, font_scale, color, thickness)

        if mouse["drawing"] or mouse["rect_ready"]:
            x1, y1 = mouse["start_x"], mouse["start_y"]
            x2, y2 = mouse["end_x"], mouse["end_y"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if mouse["rect_ready"]:
            x1, y1 = mouse["start_x"], mouse["start_y"]
            x2, y2 = mouse["end_x"], mouse["end_y"]
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            if x2 > x1 and y2 > y1:
                crop = display[y1:y2, x1:x2]
                detections = run_paddleocr(crop, ocr_engine)
                last_ocr_text = " ".join([d["text"] for d in detections]) or "No text detected"
                last_rect = [x1, y1, x2, y2]

                zoom_w = 520
                crop_h, crop_w = crop.shape[:2]
                zoom_scale = zoom_w / crop_w if crop_w > 0 else 1
                zoom_h = max(int(crop_h * zoom_scale), 50)
                zoomed = cv2.resize(crop, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Zoomed Region", zoomed)

            mouse["rect_ready"] = False

        if last_ocr_text and last_rect:
            cv2.putText(
                display,
                f"Manual: {last_ocr_text}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            display,
            f"FPS: {fps_display:.1f}",
            (10, display_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Real-Time OCR Video", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            auto_detect_mode = not auto_detect_mode
        if key == ord('c'):
            last_ocr_text = ""
            last_rect = []
            if cv2.getWindowProperty("Zoomed Region", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Zoomed Region")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
