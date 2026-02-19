import numpy as np

def crop_torso(frame: np.ndarray, xyxy: np.ndarray, torso_top: float = 0.15, torso_bottom: float = 0.65):
    """
    Crop upper body region to focus on jersey appearance.
    xyxy = (x1,y1,x2,y2) in pixels.
    """
    x1, y1, x2, y2 = xyxy.astype(int)
    h = max(1, y2 - y1)

    y1_t = y1 + int(torso_top * h)
    y2_t = y1 + int(torso_bottom * h)

    y1_t = max(0, y1_t)
    y2_t = min(frame.shape[0], y2_t)
    x1 = max(0, x1)
    x2 = min(frame.shape[1], x2)

    if y2_t <= y1_t or x2 <= x1:
        return None

    crop = frame[y1_t:y2_t, x1:x2]
    if crop.size == 0:
        return None
    return crop
