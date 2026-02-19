import numpy as np
import supervision as sv

def make_team_annotators(box_thickness: int, font_scale: float):
    palette = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])  # Team0, Team1, Other
    box = sv.BoxAnnotator(color=palette, thickness=box_thickness)
    label = sv.LabelAnnotator(color=palette, text_color=sv.Color.from_hex("#000000"), text_scale=font_scale)
    return box, label

def make_team_labels(dets: sv.Detections, show_id: bool):
    name = {0: "Team0", 1: "Team1", 2: "Other"}
    labels = []
    for i in range(len(dets)):
        tag = name.get(int(dets.class_id[i]), "Other")
        if show_id and dets.tracker_id is not None:
            labels.append(f"{tag} ID:{int(dets.tracker_id[i])}")
        else:
            labels.append(tag)
    return labels
