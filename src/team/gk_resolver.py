import numpy as np
import supervision as sv

def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)

    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team0_mask = players.class_id == 0
    team1_mask = players.class_id == 1
    if team0_mask.sum() == 0 or team1_mask.sum() == 0:
        return np.full(len(goalkeepers_xy), -1, dtype=int)

    team_0_centroid = players_xy[team0_mask].mean(axis=0)
    team_1_centroid = players_xy[team1_mask].mean(axis=0)

    out = []    # goal keepers team id
    for gxy in goalkeepers_xy:
        d0 = np.linalg.norm(gxy - team_0_centroid)
        d1 = np.linalg.norm(gxy - team_1_centroid)
        out.append(0 if d0 < d1 else 1)

    return np.array(out, dtype=int)
