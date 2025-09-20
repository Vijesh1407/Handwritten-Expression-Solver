import cv2
import numpy as np

def preprocess_and_segment(image_pil):
    img = np.array(image_pil)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15 or h < 15 or w > 150 or h > 150:
            continue
        boxes.append((x, y, w, h))

    # --- Merge overlapping/close boxes ---
    merged_boxes = []
    for box in sorted(boxes, key=lambda b: b[0]):  # sort left to right
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            prev = merged_boxes[-1]
            # If current box overlaps or is close to previous one → merge
            if box[0] < prev[0] + prev[2] + 10 and abs(box[1] - prev[1]) < 30:
                new_x = min(prev[0], box[0])
                new_y = min(prev[1], box[1])
                new_w = max(prev[0] + prev[2], box[0] + box[2]) - new_x
                new_h = max(prev[1] + prev[3], box[1] + box[3]) - new_y
                merged_boxes[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged_boxes.append(box)

    # Extract and resize symbols
    symbols = []
    for (x, y, w, h) in merged_boxes:
        roi = thresh[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        symbols.append((x, roi_resized))

    # Sort final symbols left-to-right
    symbols = sorted(symbols, key=lambda s: s[0])
    processed_symbols = [s[1] for s in symbols]

    return processed_symbols
