import cv2
import numpy as np

def preprocess_and_segment(image_pil):
    """
    Enhanced preprocessing and segmentation with better symbol detection
    Fixed to handle equals sign properly and prevent false detections
    """
    img = np.array(image_pil)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert if background is dark
    if np.mean(img) < 127:
        img = 255 - img
    
    # Crop to content area to remove excess white space
    img = crop_to_content(img)
    
    if img is None or img.size == 0:
        return []
    
    original_height, original_width = img.shape
    
    # Enhance contrast
    img = cv2.equalizeHist(img)
    
    # Strong denoising
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 10
    )
    
    # Remove very small noise
    kernel_small = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Extract bounding boxes with filtering
    boxes = []
    img_height, img_width = thresh.shape
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # More aggressive minimum area filtering
        if area < 80:  # Increased from 30
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Stricter size constraints
        if w < 8 or h < 12:  # Increased minimum size
            continue
        if w > img_width * 0.85 or h > img_height * 0.85:
            continue
        
        # Aspect ratio check - be more strict
        aspect_ratio = w / float(h)
        
        # Filter out very flat shapes (likely noise or artifacts)
        # But allow horizontal operators like "=" and "-"
        if aspect_ratio > 6:  # Very wide - likely noise or broken equals
            continue
        if aspect_ratio < 0.1:  # Very tall - likely noise
            continue
        
        # Density check - ensure there's enough content
        roi = thresh[y:y+h, x:x+w]
        density = np.sum(roi > 0) / (w * h)
        
        # More strict density check
        if density < 0.08:  # Increased from 0.05
            continue
        
        # For very wide boxes, check if they're actually equals signs
        if aspect_ratio > 2.5:
            # Check if this is a legitimate horizontal line
            if density < 0.15:  # Too sparse
                continue
        
        boxes.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'density': density
        })
    
    if not boxes:
        return []
    
    # Sort boxes by x-coordinate (left to right)
    boxes = sorted(boxes, key=lambda b: b['x'])
    
    # SMART MERGING FOR EQUALS SIGN
    merged_boxes = []
    used_indices = set()
    
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        
        box1 = boxes[i]
        merged = False
        
        # Check if this could be part of an equals sign
        # Equals sign: two horizontal lines close together vertically
        if box1['aspect_ratio'] > 1.5:  # Horizontal line
            # Look for another horizontal line nearby
            for j in range(i + 1, len(boxes)):
                if j in used_indices:
                    continue
                
                box2 = boxes[j]
                
                # Both should be horizontal
                if box2['aspect_ratio'] < 1.5:
                    continue
                
                # Calculate distances
                horizontal_gap = box2['x'] - (box1['x'] + box1['w'])
                vertical_gap = abs(box1['y'] - box2['y'])
                
                # Check if they could be equals sign components
                # They should be close horizontally and vertically aligned
                if (horizontal_gap < 25 and  # Close horizontally
                    vertical_gap < box1['h'] * 2 and  # Close vertically
                    abs(box1['w'] - box2['w']) < max(box1['w'], box2['w']) * 0.4):  # Similar width
                    
                    # Merge into equals sign
                    new_x = min(box1['x'], box2['x'])
                    new_y = min(box1['y'], box2['y'])
                    new_w = max(box1['x'] + box1['w'], box2['x'] + box2['w']) - new_x
                    new_h = max(box1['y'] + box1['h'], box2['y'] + box2['h']) - new_y
                    
                    merged_boxes.append({
                        'x': new_x, 'y': new_y, 'w': new_w, 'h': new_h,
                        'is_merged': True
                    })
                    
                    used_indices.add(i)
                    used_indices.add(j)
                    merged = True
                    break
        
        if not merged and i not in used_indices:
            merged_boxes.append(box1)
            used_indices.add(i)
    
    # Remove overlapping boxes (keep the larger/better one)
    final_boxes = []
    for i, box in enumerate(merged_boxes):
        is_duplicate = False
        
        for other_box in final_boxes:
            # Calculate overlap
            x_overlap = max(0, min(box['x'] + box['w'], other_box['x'] + other_box['w']) - 
                          max(box['x'], other_box['x']))
            y_overlap = max(0, min(box['y'] + box['h'], other_box['y'] + other_box['h']) - 
                          max(box['y'], other_box['y']))
            
            overlap_area = x_overlap * y_overlap
            box_area = box['w'] * box['h']
            other_area = other_box['w'] * other_box['h']
            
            # If significant overlap, it's a duplicate
            if overlap_area > 0.5 * min(box_area, other_area):
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_boxes.append(box)
    
    # Sort final boxes by x-coordinate
    final_boxes = sorted(final_boxes, key=lambda b: b['x'])
    
    # Additional filtering: remove boxes that are likely noise
    # by checking distance from previous box
    filtered_boxes = []
    for i, box in enumerate(final_boxes):
        if i == 0:
            filtered_boxes.append(box)
        else:
            prev_box = filtered_boxes[-1]
            gap = box['x'] - (prev_box['x'] + prev_box['w'])
            
            # If gap is too small (symbols touching), skip likely noise
            if gap < 5 and box['w'] * box['h'] < 150:  # Very small and very close
                continue
            
            filtered_boxes.append(box)
    
    # Extract, pad, and resize symbols
    symbols = []
    for box in filtered_boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        roi = thresh[y:y+h, x:x+w]
        
        # Add substantial padding
        pad = max(10, int(max(h, w) * 0.25))
        roi = cv2.copyMakeBorder(
            roi, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # Make square image (important for consistent recognition)
        h_new, w_new = roi.shape
        max_dim = max(h_new, w_new)
        
        square_roi = np.zeros((max_dim, max_dim), dtype=np.uint8)
        y_offset = (max_dim - h_new) // 2
        x_offset = (max_dim - w_new) // 2
        square_roi[y_offset:y_offset+h_new, x_offset:x_offset+w_new] = roi
        
        # Resize to 28x28 with INTER_AREA for better downsampling
        roi_resized = cv2.resize(square_roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize
        roi_resized = cv2.normalize(roi_resized, None, 0, 255, cv2.NORM_MINMAX)
        
        # Slight thickening for better recognition
        kernel = np.ones((2, 2), np.uint8)
        roi_resized = cv2.dilate(roi_resized, kernel, iterations=1)
        
        symbols.append((x, roi_resized))
    
    # Sort by x-coordinate and return only the processed images
    symbols = sorted(symbols, key=lambda s: s[0])
    processed_symbols = [s[1] for s in symbols]
    
    return processed_symbols


def crop_to_content(img):
    """
    Crop image to the bounding box of all content
    """
    # Threshold to find content
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find all non-zero points
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        return img
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add margin
    margin = 15
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)
    
    # Crop
    cropped = img[y:y+h, x:x+w]
    
    return cropped