import numpy as np
import supervision as sv

def detections_from_ultralytics(r) -> sv.Detections:
    """
    Convert Ultralytics Results (single frame) to sv.Detections, preserving tracker_id if present.
    """
    if r.boxes is None or len(r.boxes) == 0:
        return sv.Detections.empty()

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls  = r.boxes.cls.cpu().numpy().astype(int)

    tid = None
    if getattr(r.boxes, "id", None) is not None and r.boxes.id is not None:
        tid = r.boxes.id.cpu().numpy().astype(int)

    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls, tracker_id=tid)
