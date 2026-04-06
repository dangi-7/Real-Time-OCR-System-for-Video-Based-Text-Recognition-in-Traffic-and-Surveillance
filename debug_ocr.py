import os
os.environ.setdefault("FLAGS_use_pir", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2, time
from paddleocr import PaddleOCR


print('Loading PaddleOCR...')
ocr_engine = PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    device='cpu',
    enable_mkldnn=False,
    enable_cinn=False,
)
print('Loaded!')

cap = cv2.VideoCapture('sample_video.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Total frames: {total}')

all_results = []
for i in range(0, min(total, 300), 30):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break

    t = time.time()
    results = ocr_engine.predict(
        frame,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1280,
        text_det_limit_type="max",
        text_det_thresh=0.20,
        text_det_box_thresh=0.30,
        text_det_unclip_ratio=1.6,
        text_rec_score_thresh=0.20,
    )
    dt = time.time() - t

    texts = []
    if results:
        for res in results:
            if not isinstance(res, dict):
                continue
            texts_list = res.get("rec_texts") or []
            scores_list = res.get("rec_scores") or []
            for text_item, conf in zip(texts_list, scores_list):
                text = text_item[0] if isinstance(text_item, (list, tuple)) else text_item
                text = str(text).strip()
                if text:
                    texts.append(f"{text}({float(conf):.2f})")

    line = f"Frame {i}: {len(texts)} texts in {dt:.1f}s: {', '.join(texts)}"
    print(line)
    all_results.append(line)

cap.release()

with open('ocr_test_output.txt', 'w') as f:
    f.write('\n'.join(all_results))
print('\nResults saved to ocr_test_output.txt')
print('DONE')
