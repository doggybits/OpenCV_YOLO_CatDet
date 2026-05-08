# cat_color_classifier.py
import cv2
import numpy as np

# Import the debug module (it won't break anything if not used)
try:
    from cat_color_debug import CatColorDebugger
except ImportError:
    CatColorDebugger = None


def classify_cat_color(cat_crop, debug=False):
    """
    Classifies cat as Orange or White.
    
    Parameters:
        cat_crop: numpy array (BGR image crop)
        debug: bool - whether to show debug info
    
    Returns:
        label, box_color, confidence
    """

    if cat_crop is None or cat_crop.size == 0:
        return "Unknown", (128, 128, 128), 0.0

    # Resize for consistency
    small = cv2.resize(cat_crop, (160, 160))

    # Center crop (body focus)
    h, w = small.shape[:2]
    center = small[h//4:3*h//4, w//4:3*w//4]

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    # ================== MASKS ==================
    lower_orange1 = np.array([5,  70,  65])
    upper_orange1 = np.array([25, 255, 255])
    lower_orange2 = np.array([0,  70,  65])
    upper_orange2 = np.array([10, 255, 255])

    orange_mask = (cv2.inRange(hsv, lower_orange1, upper_orange1) | 
                   cv2.inRange(hsv, lower_orange2, upper_orange2))

    lower_white = np.array([0,   0,  165])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # ================== RATIOS ==================
    total = center.shape[0] * center.shape[1]
    orange_ratio = cv2.countNonZero(orange_mask) / total
    white_ratio = cv2.countNonZero(white_mask) / total

    # ================== CLASSIFICATION ==================
    if orange_ratio > 0.33 and brightness < 230:
        label = "Orange Cat"
        box_color = (0, 165, 255)
        confidence = orange_ratio

    elif white_ratio > 0.40 or brightness > 235:
        label = "White Cat"
        box_color = (255, 255, 255)
        confidence = max(white_ratio, (brightness - 200) / 60)

    elif orange_ratio > white_ratio + 0.12:
        label = "Orange Cat"
        box_color = (0, 165, 255)
        confidence = orange_ratio
    else:
        label = "White Cat"
        box_color = (255, 255, 255)
        confidence = white_ratio

    confidence = min(0.98, float(confidence))

    # ================== DEBUG ==================
    if debug and CatColorDebugger is not None:
        CatColorDebugger.show_debug_info(
            center, orange_mask, white_mask,
            orange_ratio, white_ratio, brightness,
            label, confidence
        )

    return label, box_color, confidence