# %%
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from skimage import morphology

# ============================================================
#  WSPÓLNE NARZĘDZIA — używane przez wszystkie moduły
# ============================================================

def get_roi(img):
    """
    Wycina kabel kołową maską (jak w podejściu wzorcowym).
    Zwraca: roi_bgr, (cx, cy, radius), thresh_val
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 15)
    tv, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cx, cy, radius = int(cx), int(cy), int(radius)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    roi = cv2.bitwise_and(img, img, mask=mask)
    return roi, (cx, cy, radius), tv


def get_color_masks(roi_image, roi_mask):
    """
    Segmentacja 3 żył kabla w przestrzeni HSV.
    Zwraca słownik: {'blue': mask, 'brown': mask, 'yg': mask}
    Identyczny z podejściem wzorcowym — używany przez wszystkie moduły.
    """
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks = {}

    lower_blue  = np.array([90, 130, 20]);  upper_blue  = np.array([130, 255, 255])
    lower_yg    = np.array([25, 80,  20]);  upper_yg    = np.array([85,  255, 255])
    lower_brown = np.array([10, 0,   0]);   upper_brown = np.array([30,  30,  100])

    raw_set = [('blue', lower_blue, upper_blue),
               ('brown', lower_brown, upper_brown),
               ('yg', lower_yg, upper_yg)]

    for name, lo, hi in raw_set:
        raw = cv2.inRange(hsv, lo, hi)
        raw = cv2.bitwise_and(raw, roi_mask)
        m   = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k_close)
        masks[name] = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    return masks


# ============================================================
#  MODUŁ 1 — CUT INNER INSULATION
#  Wzorzec: convexityDefects na każdej żyle osobno
# ============================================================

INNER_CUT_DEPTH_THRESHOLD = 5.0  # px (żyły są mniejsze niż cały kabel)

def detect_cuts(img):
    """
    Naprawa: zamiast CLAHE + HSV + analiza kołowości, używamy convexityDefects
    na maskach kolorowych żył — dokładnie jak w module wzorcowym.
    """
    roi, (cx, cy, r), _ = get_roi(img)
    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), r, 255, -1)

    color_masks = get_color_masks(roi, roi_mask)
    defect_mask = np.zeros(roi.shape[:2], dtype=np.uint8)

    for name, wire_mask in color_masks.items():
        if cv2.countNonZero(wire_mask) < 500:
            continue
        contours, _ = cv2.findContours(wire_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            continue
        eps    = 0.002 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True)
        hull   = cv2.convexHull(approx, returnPoints=False)
        try:
            defects = cv2.convexityDefects(approx, hull)
        except cv2.error:
            continue
        if defects is None:
            continue
        for d in defects:
            s, e, f, depth_raw = d[0]
            if depth_raw / 256.0 > INNER_CUT_DEPTH_THRESHOLD:
                pts = np.array([approx[s][0], approx[f][0], approx[e][0]])
                cv2.fillPoly(defect_mask, [pts], 255)

    return defect_mask


def process_all_images_cut_inner(input_folder, gt_folder):
    images = sorted([f for f in os.listdir(input_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))
    if len(images) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, filename in enumerate(images):
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue
        result_mask = detect_cuts(img)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Oryginał: {filename}")
        axes[i, 1].imshow(result_mask, cmap='gray')
        axes[i, 1].set_title("Wykryte nacięcia (convexityDefects)")

        name_part = os.path.splitext(filename)[0]
        gt_path = os.path.join(gt_folder, f"{name_part}_mask.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            _, gt_bin = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
            axes[i, 2].imshow(gt_bin, cmap='gray')
        axes[i, 2].set_title("Maska GT")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
#  MODUŁ 2 — MISSING CABLE
#  Wzorzec: porównanie pola żył z medianą → estymacja pozycji przez symetrię
# ============================================================

MISSING_CABLE_RATIO = 0.15

def detect_missing_cable(img, n_samples=1):
    """
    Naprawa: zamiast globalnego progowania + morfologii, używamy masek HSV.
    Porównujemy pole każdej żyły z medianą. Brakująca < 15% mediany.
    Pozycję szacujemy przez symetrię względem centrum kabla.
    """
    roi, (cx, cy, r), _ = get_roi(img)
    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), r, 255, -1)

    color_masks = get_color_masks(roi, roi_mask)
    areas = {n: cv2.countNonZero(m) for n, m in color_masks.items()}
    median_area = float(np.median(list(areas.values())))

    defect_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    if median_area < 100:
        return defect_mask, False

    missing, present = [], {}
    for name, area in areas.items():
        if area < median_area * MISSING_CABLE_RATIO:
            missing.append(name)
        else:
            M = cv2.moments(color_masks[name])
            if M['m00'] > 0:
                present[name] = (M['m10'] / M['m00'], M['m01'] / M['m00'])

    has_defect = len(missing) > 0
    if has_defect and len(present) >= 2:
        pres_cx = np.mean([p[0] for p in present.values()])
        pres_cy = np.mean([p[1] for p in present.values()])
        miss_x  = int(2 * cx - pres_cx)
        miss_y  = int(2 * cy - pres_cy)
        est_r   = int(np.sqrt(median_area / math.pi))
        cv2.circle(defect_mask, (miss_x, miss_y), est_r, 255, -1)

    return defect_mask, has_defect


def run_missing_cable(n_samples=3):
    input_dir = os.path.join('cable', 'test', 'missing_cable')
    gt_dir    = os.path.join('cable', 'ground_truth', 'missing_cable')

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    samples = image_files[:n_samples]

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, img_name in enumerate(samples):
        img = cv2.imread(os.path.join(input_dir, img_name))
        if img is None:
            continue

        result_mask, has_defect = detect_missing_cable(img)
        color_title = 'red' if has_defect else 'green'
        status      = "BŁĄD: BRAK ŻYŁY!" if has_defect else "STATUS: OK"

        mask_name = img_name.replace('.png', '_mask.png')
        gt_mask   = cv2.imread(os.path.join(gt_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original: {img_name}", fontsize=10)
        axes[i, 1].imshow(result_mask, cmap='gray')
        axes[i, 1].set_title(status, color=color_title, fontsize=10)
        axes[i, 2].imshow(gt_mask, cmap='gray')
        axes[i, 2].set_title("Original Ground Truth", fontsize=10)

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
#  MODUŁ 3 — CUT OUTER INSULATION
#  Wzorzec: convexHull + convexityDefects na konturze całego kabla
# ============================================================

OUTER_CUT_DEPTH_THRESHOLD = 100.0  # px

def detect_outer_cut(img):
    """
    Naprawa: zamiast morphological reconstruction (skimage), używamy
    convexityDefects na konturze całego kabla — głębokość > 100 px = nacięcie.
    """
    roi, _, _ = get_roi(img)
    defect_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return defect_mask, False

    largest = max(contours, key=cv2.contourArea)
    eps     = 0.001 * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, eps, True)  # KRYTYCZNE: zapobiega crashowi
    hull    = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    has_defect = False
    if defects is not None:
        for d in defects:
            s, e, f, depth_raw = d[0]
            depth = depth_raw / 256.0
            if depth > OUTER_CUT_DEPTH_THRESHOLD:
                has_defect = True
                pts = np.array([approx[s][0], approx[f][0], approx[e][0]])
                cv2.fillPoly(defect_mask, [pts], 255)

    return defect_mask, has_defect


def process_all_images_cut_outer(input_folder, gt_folder):
    images = sorted([f for f in os.listdir(input_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))
    if len(images) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, filename in enumerate(images):
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue

        defect_mask, has_defect = detect_outer_cut(img)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Oryginał: {filename}")
        axes[i, 1].imshow(defect_mask, cmap='gray')
        axes[i, 1].set_title(f"Nacięcie zewn. (has={has_defect})")

        name_part = os.path.splitext(filename)[0]
        gt_path   = os.path.join(gt_folder, f"{name_part}_mask.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            _, gt_bin = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
            axes[i, 2].imshow(gt_bin, cmap='gray')
        axes[i, 2].set_title("Maska GT")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
#  MODUŁ 4 — MISSING WIRE (brak rdzenia miedzianego)
#  Wzorzec: wypełnij kontur żyły → odejmij maskę → sprawdź ile miedzi w dziurze
# ============================================================

COPPER_BRIGHT_THRESH = 200  # px jasności = miedź

def detect_missing_wire(img):
    """
    Naprawa: zamiast szukać ciemnych plam wewnątrz okręgu, prawidłowo:
    1. Wypełniamy każdy kontur żyły (solid_wire)
    2. Dziura = solid_wire - wire_mask
    3. Jeśli miedź zajmuje < 20% dziury → brak rdzenia
    """
    roi, (cx, cy, r), _ = get_roi(img)
    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), r, 255, -1)

    color_masks = get_color_masks(roi, roi_mask)
    defect_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    has_defect  = False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, copper = cv2.threshold(gray, COPPER_BRIGHT_THRESH, 255, cv2.THRESH_BINARY)

    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for name, wmask in color_masks.items():
        contours, _ = cv2.findContours(wmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest_cnt = max(contours, key=cv2.contourArea)
        solid_wire  = np.zeros_like(wmask)
        cv2.drawContours(solid_wire, [largest_cnt], -1, 255, thickness=cv2.FILLED)

        # Dziura = wypełniony kontur MINUS maska koloru (środek żyły gdzie powinna być miedź)
        hole_mask = cv2.bitwise_and(solid_wire, cv2.bitwise_not(wmask))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, k_clean)

        hole_area = cv2.countNonZero(hole_mask)
        if hole_area < 1000:
            continue

        copper_in_hole = cv2.bitwise_and(copper, hole_mask)
        copper_area    = cv2.countNonZero(copper_in_hole)

        # Jeśli miedź zajmuje < 20% dziury → dziura jest pusta = brak rdzenia
        if (copper_area / hole_area) < 0.20:
            has_defect = True
            cv2.bitwise_or(defect_mask, hole_mask, defect_mask)

    return defect_mask, has_defect


def process_missing_wire_dataset(input_folder, gt_folder):
    images = sorted([f for f in os.listdir(input_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        print(f"Pusty folder: {input_folder}")
        return

    fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))
    if len(images) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, filename in enumerate(images):
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue

        result_mask, has_missing = detect_missing_wire(img)
        color_title = 'red'   if has_missing else 'green'
        status      = "BŁĄD: BRAK ŻYŁY!" if has_missing else "STATUS: OK"

        name_part = os.path.splitext(filename)[0]
        gt_path   = os.path.join(gt_folder, f"{name_part}_mask.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            gt = np.zeros(img.shape[:2], dtype=np.uint8)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original: {filename}")
        axes[i, 1].imshow(result_mask, cmap='gray')
        axes[i, 1].set_title(status, color=color_title)
        axes[i, 2].imshow(gt, cmap='gray')
        axes[i, 2].set_title(f"GT: {filename}")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
#  MODUŁ 5 — BENT WIRE
#  Wzorzec: kołowość (circularity) konturu miedzianych obszarów < 0.60
# ============================================================

def detect_bent_wire(img):
    """
    Naprawa: zamiast odejmować "dobre" rdzenie od całej miedzi,
    używamy kołowości (circularity) — zagięty rdzeń ma nieregularny kształt.
    Eliminujemy też zewnętrzny biały pierścień przez erozję maski ROI.
    """
    roi, (cx, cy, r), _ = get_roi(img)
    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), r, 255, -1)

    defect_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    gray        = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Eliminacja zewnętrznego pierścienia izolacji przez skurczenie maski
    k_erode   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    safe_zone = cv2.erode(roi_mask, k_erode)

    # Miedź = jasne obszary wewnątrz bezpiecznej strefy
    _, copper = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    copper    = cv2.bitwise_and(copper, safe_zone)

    # Sklejamy druciki w jeden rdzeń
    k_fuse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    copper = cv2.morphologyEx(copper, cv2.MORPH_CLOSE, k_fuse)

    contours, _ = cv2.findContours(copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_defect  = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        # Prawidłowy rdzeń: circularity ~0.8. Zagięty: < 0.60
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        if circularity < 0.60:
            has_defect = True
            cv2.drawContours(defect_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return defect_mask, has_defect


def process_bent_wire_visual(input_folder, gt_folder):
    images = sorted([f for f in os.listdir(input_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        print(f"Brak zdjęć w {input_folder}")
        return

    fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))
    if len(images) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, filename in enumerate(images):
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue

        raw_mask, has_defect = detect_bent_wire(img)

        gt_filename = f"{os.path.splitext(filename)[0]}_mask.png"
        gt = cv2.imread(os.path.join(gt_folder, gt_filename), cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original: {filename}")
        axes[i, 1].imshow(raw_mask, cmap='gray')
        axes[i, 1].set_title(f"Bent wire mask (has={has_defect})")

        if gt is not None:
            _, gt_bin = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
            axes[i, 2].imshow(gt_bin, cmap='gray')
            axes[i, 2].set_title("Ground Truth")
        else:
            axes[i, 2].imshow(np.zeros_like(raw_mask), cmap='gray')
            axes[i, 2].set_title("GT Missing")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
#  MODUŁ 6 — CABLE SWAP
#  Wzorzec: K-Means na pikselach żył → 3 klastry → kąt + kolejność kolorów
# ============================================================

EXPECTED_ORDER = ['yg', 'brown', 'blue']

def detect_cable_swap(img):
    """
    Naprawa: zamiast HSV + morfologii + skiz_for_donuts + analiza liter,
    używamy K-Means żeby podzielić żyły na 3 klastry, sprawdzamy ich
    kątową kolejność względem centrum kabla.
    """
    roi, (cx, cy, r), _ = get_roi(img)
    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), r, 255, -1)

    color_masks = get_color_masks(roi, roi_mask)
    defect_mask = np.zeros(roi.shape[:2], dtype=np.uint8)

    # Łączymy wszystkie maski żył
    combined_mask = np.zeros_like(roi_mask)
    for m in color_masks.values():
        combined_mask = cv2.bitwise_or(combined_mask, m)

    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, k_clean)

    y_coords, x_coords = np.where(combined_mask > 0)
    if len(x_coords) < 5000:
        return defect_mask, False

    points   = np.column_stack((x_coords, y_coords)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(points, 3, None, criteria, 10,
                                    cv2.KMEANS_PP_CENTERS)

    wires_info = []
    for i in range(3):
        center_x, center_y = centers[i]
        angle = math.atan2(center_y - cy, center_x - cx)

        cluster_mask = np.zeros_like(roi_mask)
        cluster_pts  = points[labels.flatten() == i].astype(np.int32)
        cluster_mask[cluster_pts[:, 1], cluster_pts[:, 0]] = 255
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, k_clean)

        best_color, max_overlap = None, 0
        for color_name, c_mask in color_masks.items():
            overlap = cv2.countNonZero(cv2.bitwise_and(cluster_mask, c_mask))
            if overlap > max_overlap:
                max_overlap   = overlap
                best_color    = color_name

        wires_info.append({'angle': angle, 'color': best_color, 'mask': cluster_mask})

    wires_info.sort(key=lambda x: x['angle'])
    actual_colors = [w['color'] for w in wires_info]

    best_shift, min_mismatches = 0, 4
    for shift in range(3):
        shifted   = EXPECTED_ORDER[shift:] + EXPECTED_ORDER[:shift]
        mismatches = sum(1 for j in range(3) if actual_colors[j] != shifted[j])
        if mismatches < min_mismatches:
            min_mismatches = mismatches
            best_shift     = shift

    has_defect = False
    if min_mismatches > 0:
        has_defect     = True
        best_expected  = EXPECTED_ORDER[best_shift:] + EXPECTED_ORDER[:best_shift]
        for j in range(3):
            if actual_colors[j] != best_expected[j]:
                cv2.bitwise_or(defect_mask, wires_info[j]['mask'], defect_mask)

    return defect_mask, has_defect


def visualize_cable_swap(input_folder, gt_folder):
    images = sorted([f for f in os.listdir(input_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        return

    fig, axes = plt.subplots(len(images), 4, figsize=(20, 4 * len(images)))
    if len(images) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, filename in enumerate(images):
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue

        roi, (cx, cy, r), _ = get_roi(img)
        roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(roi_mask, (cx, cy), r, 255, -1)
        color_masks = get_color_masks(roi, roi_mask)

        defect_mask, has_defect = detect_cable_swap(img)

        color_vis = np.zeros((*roi.shape[:2], 3), dtype=np.uint8)
        color_vis[color_masks['blue']  > 0] = [0, 100, 220]
        color_vis[color_masks['brown'] > 0] = [160, 80, 30]
        color_vis[color_masks['yg']    > 0] = [80, 200, 50]

        gt_path = os.path.join(gt_folder, f"{os.path.splitext(filename)[0]}_mask.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Oryginał: {filename}")
        axes[i, 1].imshow(color_vis)
        axes[i, 1].set_title("Wykryte kolory żył")
        axes[i, 2].imshow(defect_mask, cmap='gray')
        axes[i, 2].set_title(f"Maska wady (has={has_defect})")
        if gt is not None:
            _, gt_bin = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
            axes[i, 3].imshow(gt_bin, cmap='gray')
        else:
            axes[i, 3].imshow(np.zeros_like(defect_mask), cmap='gray')
        axes[i, 3].set_title("GT maska")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
#  MODUŁ 7 — POKE INSULATION
#  Wzorzec: izoluje pierścień izolacji, szuka anomalii jasności i gradientu
# ============================================================

def detect_poke_insulation(img):
    """
    Naprawa: zamiast K-Means globalnego + maski koniczynki,
    izolujemy pierścień izolacji (ROI minus żyły minus miedź),
    a następnie szukamy w nim anomalii przez lokalną jasność i gradient.
    """
    roi, (cx, cy, r), _ = get_roi(img)
    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), r, 255, -1)

    color_masks = get_color_masks(roi, roi_mask)
    h, w        = roi.shape[:2]
    gray        = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    defect_mask = np.zeros((h, w), dtype=np.uint8)

    # Krok 1: usuń żyły kolorowe + ich otoczenie
    k_big      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
    wire_union = np.zeros((h, w), dtype=np.uint8)
    for mask in color_masks.values():
        wire_union = cv2.bitwise_or(wire_union, mask)
    wire_big = cv2.dilate(wire_union, k_big, iterations=2)

    # Krok 2: miedź przez lokalny kontrast (a nie globalny próg)
    gray_blur       = cv2.GaussianBlur(gray, (31, 31), 0)
    local_highlight = np.clip(gray.astype(np.int16) - gray_blur.astype(np.int16),
                              0, 255).astype(np.uint8)
    _, copper_mask  = cv2.threshold(local_highlight, 40, 255, cv2.THRESH_BINARY)
    k_copper        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    copper_mask     = cv2.dilate(copper_mask, k_copper, iterations=2)
    copper_mask     = cv2.bitwise_and(copper_mask, wire_big)

    # Krok 3: strefa wykluczona = żyły + miedź
    exclude_mask = cv2.bitwise_or(wire_big, copper_mask)

    # Krok 4: pierścień izolacji = ROI minus wykluczone
    insulation_ring = cv2.bitwise_and(roi_mask, cv2.bitwise_not(exclude_mask))
    _, not_background = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    insulation_ring   = cv2.bitwise_and(insulation_ring, not_background)

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    insulation_ring = cv2.morphologyEx(insulation_ring, cv2.MORPH_OPEN,  k7)
    insulation_ring = cv2.morphologyEx(insulation_ring, cv2.MORPH_CLOSE, k7)

    insulation_pixels = gray[insulation_ring > 0]
    if len(insulation_pixels) < 100:
        return defect_mask, False

    # Krok 5: anomalia jasności względem mediany pierścienia
    ref      = np.median(insulation_pixels).astype(np.float32)
    dark_diff = np.clip(ref - gray.astype(np.float32), 0, 255).astype(np.uint8)
    _, dark_bin = cv2.threshold(dark_diff, 25, 255, cv2.THRESH_BINARY)

    # Krok 6: anomalia tekstury (gradient Sobela)
    gray_masked = cv2.bitwise_and(gray, insulation_ring)
    sobelx = cv2.Sobel(gray_masked, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_masked, cv2.CV_64F, 0, 1, ksize=3)
    grad   = np.sqrt(sobelx**2 + sobely**2)
    grad_norm = np.clip(grad / (grad.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    grad_blur = cv2.GaussianBlur(grad_norm, (9, 9), 0)
    _, grad_bin = cv2.threshold(grad_blur, 40, 255, cv2.THRESH_BINARY)

    # Krok 7: kombinacja tylko w pierścieniu izolacji
    combined = cv2.bitwise_or(
        cv2.bitwise_and(dark_bin, insulation_ring),
        cv2.bitwise_and(grad_bin, insulation_ring)
    )

    # Krok 8: morfologia końcowa + filtracja małych obiektów
    k9  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,  9))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k9)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k15)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, 8)
    for j in range(1, n_labels):
        if stats[j, cv2.CC_STAT_AREA] >= 150:
            defect_mask[labels == j] = 255

    has_defect = cv2.countNonZero(defect_mask) > 0
    return defect_mask, has_defect


def process_with_poke_logic(input_folder, gt_folder):
    images = sorted([f for f in os.listdir(input_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        return

    fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))
    if len(images) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, filename in enumerate(images):
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue

        defect_mask, has_defect = detect_poke_insulation(img)

        gt_path = os.path.join(gt_folder, f"{os.path.splitext(filename)[0]}_mask.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Oryginał: {filename}")
        axes[i, 1].imshow(defect_mask, cmap='gray')
        axes[i, 1].set_title(f"Wkłucie w izolację (has={has_defect})")
        if gt is not None:
            axes[i, 2].imshow(gt, cmap='gray')
            axes[i, 2].set_title("Ground Truth")
        else:
            axes[i, 2].imshow(np.zeros_like(defect_mask), cmap='gray')
            axes[i, 2].set_title("GT brak")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
#  WYWOŁANIA — odkomentuj interesującą kategorię
# ============================================================

if __name__ == '__main__':

    # --- CUT INNER INSULATION ---
    process_all_images_cut_inner(
        'cable/test/cut_inner_insulation',
        'cable/ground_truth/cut_inner_insulation'
    )

    # --- MISSING CABLE ---
    run_missing_cable(n_samples=3)

    # --- CUT OUTER INSULATION ---
    process_all_images_cut_outer(
        'cable/test/cut_outer_insulation',
        'cable/ground_truth/cut_outer_insulation'
    )

    # --- MISSING WIRE ---
    process_missing_wire_dataset(
        'cable/test/missing_wire',
        'cable/ground_truth/missing_wire'
    )

    # --- BENT WIRE ---
    process_bent_wire_visual(
        'cable/test/bent_wire',
        'cable/ground_truth/bent_wire'
    )

    # --- CABLE SWAP ---
    visualize_cable_swap(
        'cable/test/cable_swap',
        'cable/ground_truth/cable_swap'
    )

    # --- POKE INSULATION ---
    process_with_poke_logic(
        'cable/test/poke_insulation',
        'cable/ground_truth/poke_insulation'
    )
# %%
