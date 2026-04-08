
import numpy as np
import cv2

def detect_missing_cable(img):
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5,5), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=20)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    return processed

def detect_missing_wire(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, black_areas = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)

    mask = np.zeros_like(gray)
    h, w = gray.shape
    cv2.circle(mask, (w // 2, h // 2), int(min(w, h) * 0.35), 255, -1)
    candidate_mask = cv2.bitwise_and(black_areas, mask)

    kernel_dil = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(candidate_mask, kernel_dil, iterations=1)
    
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(dilated_mask)
    
    circularity_threshold = 0.35  # Podniesione z 0.35 (bardziej "kołowe")
    solidity_threshold = 0.25     # Nowość: jak bardzo kształt jest "pełny" (brak wcięć)
    area_threshold = 3500         # Lekko podniesione

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0: continue

        circularity = 4 * np.pi * (area / (peri * peri))

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        if circularity >= circularity_threshold and solidity >= solidity_threshold:
            
          
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
    final_mask = cv2.dilate(final_mask, kernel_dil, iterations=3)

    return final_mask

def detect_bent_wire(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_copper = np.array([0, 40, 60])
    upper_copper = np.array([25, 255, 255])
    copper_mask = cv2.inRange(hsv, lower_copper, upper_copper)

    kernel_small = np.ones((3, 3), np.uint8)
    copper_mask = cv2.morphologyEx(copper_mask, cv2.MORPH_OPEN, kernel_small)
    copper_closed = cv2.morphologyEx(copper_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(copper_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good_cores_mask = np.zeros_like(copper_mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1600:
            continue
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        if solidity > 0.85 and area > 2500:
            cv2.drawContours(good_cores_mask, [cnt], -1, 255, -1)

    good_cores_mask = cv2.dilate(good_cores_mask, np.ones((7, 7), np.uint8))

    bent_only = cv2.bitwise_and(copper_mask, cv2.bitwise_not(good_cores_mask))

    final_bent_contours, _ = cv2.findContours(bent_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(bent_only)

    for cnt in final_bent_contours:
        if cv2.contourArea(cnt) > 2000:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    final_mask = cv2.dilate(clean_mask, np.ones((5, 5), np.uint8), iterations=4)

    return final_mask

def process(gt):
    if (gt == 0).all():
        return None
    
    return gt

def predict(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image: tablica NumPy, kształt (H, W, 3), dtype uint8, RGB

    Returns:
        Maska binarna, kształt (H, W), dtype uint8, wartości 0 lub 255
        255 = wada, 0 = brak wady
    """

    gt = detect_missing_cable(image)

    if process(image) is None:
        return gt
    
    gt = detect_missing_wire(image)

    if process(image) is None:
        return gt

    gt = detect_bent_wire(image)

    if process(image) is None:
        return gt




