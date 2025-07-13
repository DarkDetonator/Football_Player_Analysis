
import sys
sys.path.insert(0, r"e:\Football_tracking\ByteTrack")

import numpy as np
np.float = float
import cv2
from yolox.tracker.byte_tracker import BYTETracker
from myconfig import VIDEO_PATH, PLAYER_MODEL_PATH, BALL_MODEL_PATH, PITCH_MODEL_PATH, GOALPOST_MODEL_PATH

# Load YOLOv8 models (Ultralytics)
from ultralytics import YOLO
from config import TACTICAL_PITCH_IMAGE

# Convert detections to ByteTrack format: [x1, y1, x2, y2, score]
def yolo_to_bytetrack(results):
    # Returns [x1, y1, x2, y2, score, class_idx] for each detection
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy()
        if hasattr(results.boxes, 'cls'):
            cls = results.boxes.cls.cpu().numpy()
        else:
            cls = np.zeros(len(xyxy))
        return [
            [float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), int(c)]
            for b, s, c in zip(xyxy, conf, cls)
        ]
    else:
        return []

import myconfig
from ROI_maping.roi_mapper import project_point
abbreviation_to_number = {
    "TLC": "1",
    "TRC": "6",
    "TR6MC": "8",
    "TL6MC": "7",
    "TR6ML": "4",
    "TL6ML": "3",
    "TR18MC": "9",
    "TL18MC": "12",
    "TR18ML": "5",
    "TL18ML": "2",
    "TRArc": "10",
    "TLArc": "11",
    "RML": "16",
    "RMC": "15",
    "LMC": "14",
    "LML": "13",
    "BLC": "23",
    "BRC": "28",
    "BR6MC": "22",
    "BL6MC": "21",
    "BR6ML": "26",
    "BL6ML": "25",
    "BR18MC": "20",
    "BL18MC": "17",
    "BR18ML": "27",
    "BL18ML": "24",
    "BRArc": "19",
    "BLArc": "18"
}
TACTICAL_IMAGE = cv2.imread(myconfig.TACTICAL_PITCH_IMAGE)
import json
with open(myconfig.PITCH_LABELS_JSON, 'r') as f:
    pitch_coords = json.load(f)
if TACTICAL_IMAGE is None:
    raise FileNotFoundError(f"Tactical pitch image not found at {TACTICAL_PITCH_IMAGE}")
TACTICAL_H, TACTICAL_W = TACTICAL_IMAGE.shape[:2]

# Use central ROI region definitions
from ROI_maping.roi_definitions import ROI_REGIONS

player_model = YOLO(PLAYER_MODEL_PATH)
ball_model = YOLO(BALL_MODEL_PATH)
pitch_model = YOLO(PITCH_MODEL_PATH)
goalpost_model = YOLO(GOALPOST_MODEL_PATH)


# Create args object for BYTETracker with custom settings
class Args:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, mot20=False):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.mot20 = mot20

# Use default settings for players
player_tracker = BYTETracker(Args(track_buffer=30))
# Use lower buffer and threshold for ball and goalposts
ball_tracker = BYTETracker(Args(track_buffer=5, track_thresh=0.3))
goalpost_tracker = BYTETracker(Args(track_buffer=5, track_thresh=0.3))

# DEBUG: Print pitch_model.names if available
if hasattr(pitch_model, 'names'):
    print("[DEBUG] pitch_model.names:", pitch_model.names)
else:
    print("[DEBUG] pitch_model has no 'names' attribute.")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Failed to open video: {VIDEO_PATH}")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_h, img_w = frame.shape[:2]
    img_size = (img_h, img_w)
    info_imgs = np.array([img_h, img_w])

    # Run detection for each object type
    player_results = player_model(frame, verbose=False)[0]
    ball_results = ball_model(frame, verbose=False)[0]
    goalpost_results = goalpost_model(frame, verbose=False)[0]
    pitch_results = pitch_model(frame, verbose=False)[0]

    players = yolo_to_bytetrack(player_results)
    ball = yolo_to_bytetrack(ball_results)
    goalposts = yolo_to_bytetrack(goalpost_results)
    pitch = yolo_to_bytetrack(pitch_results)

    # Visualize all raw goalpost detections (magenta, before tracking)
    for box in goalposts:
        if len(box) > 5:
            x1, y1, x2, y2, score, *_ = box
        else:
            x1, y1, x2, y2, score = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
        cv2.putText(frame, f"GoalRaw", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Track goalposts and draw tracked ID/circle immediately after raw detections
    goalpost_dets = np.array(goalposts, dtype=np.float32)
    if goalpost_dets.size == 0:
        goalpost_dets = np.zeros((0, 5), dtype=np.float32)
    online_targets_goalposts = goalpost_tracker.update(goalpost_dets, info_imgs, img_size)
    for t in online_targets_goalposts:
        x, y, w, h = t.tlwh
        cx, cy = int(x + w/2), int(y + h/2)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        # Draw tracking ID above the box
        cv2.putText(frame, f"G{t.track_id}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # Draw a circle at the center and show the ID for easier debugging
        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), 2)
        cv2.putText(frame, f"ID:{t.track_id}", (cx+12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Visualize pitch detections (yellow boxes)
    for box in pitch:
        if len(box) > 5:
            x1, y1, x2, y2, score, *_ = box
        else:
            x1, y1, x2, y2, score = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        # Show pitch class (abbreviation) and label number if available
        label_idx = int(box[5]) if len(box) > 5 else None
        abbr = pitch_model.names[label_idx] if label_idx is not None and hasattr(pitch_model, 'names') and label_idx in pitch_model.names else str(label_idx) if label_idx is not None else "?"
        label_num = abbreviation_to_number.get(abbr, abbr)
        cv2.putText(frame, f"{abbr} ({label_num})", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    # Track players (only once per frame)
    player_dets = np.array(players, dtype=np.float32)
    if player_dets.size == 0:
        player_dets = np.zeros((0, 5), dtype=np.float32)

    online_targets_players = player_tracker.update(player_dets, info_imgs, img_size)
    print(f"[DEBUG] Player tracks: {len(online_targets_players)} | IDs: {[t.track_id for t in online_targets_players]}")
    for t in online_targets_players:
        x, y, w, h = t.tlwh
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"P{t.track_id}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Track ball (only once per frame)
    ball_dets = np.array(ball, dtype=np.float32)
    if ball_dets.size == 0:
        ball_dets = np.zeros((0, 5), dtype=np.float32)

    online_targets_ball = ball_tracker.update(ball_dets, info_imgs, img_size)
    print(f"[DEBUG] Ball tracks: {len(online_targets_ball)} | IDs: {[t.track_id for t in online_targets_ball]}")
    for t in online_targets_ball:
        x, y, w, h = t.tlwh
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.putText(frame, f"B{t.track_id}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Track goalposts
    goalpost_dets = np.array(goalposts, dtype=np.float32)
    if goalpost_dets.size == 0:
        goalpost_dets = np.zeros((0, 5), dtype=np.float32)

    online_targets_goalposts = goalpost_tracker.update(goalpost_dets, info_imgs, img_size)
    print(f"[DEBUG] Goalpost tracks: {len(online_targets_goalposts)} | IDs: {[t.track_id for t in online_targets_goalposts]}")
    for t in online_targets_goalposts:
        x, y, w, h = t.tlwh
        cx, cy = int(x + w/2), int(y + h/2)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        # Draw tracking ID above the box
        cv2.putText(frame, f"G{t.track_id}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # Draw a circle at the center and show the ID for easier debugging
        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), 2)
        cv2.putText(frame, f"ID:{t.track_id}", (cx+12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Homography projection for all objects onto tactical pitch image
    tactical_map = TACTICAL_IMAGE.copy()

    # --- MIRRORED PITCH POINT MAJORITY LOGIC ---
    mirrored_pairs = [("7","21"), ("8","22"), ("10","18"), ("11","19"), ("9","17"), ("12","20"), ("3","25"), ("4","26"), ("2","24"), ("5","27")]
    groupA = set([a for a,b in mirrored_pairs])
    groupB = set([b for a,b in mirrored_pairs])
    detected_labels = []
    label_to_det = {}
    for b in pitch:
        if len(b) > 5:
            label_idx = int(b[5])
            label = str(pitch_model.names[label_idx]) if hasattr(pitch_model, 'names') and label_idx in pitch_model.names else str(label_idx)
        else:
            label = None
        if label:
            detected_labels.append(label)
            label_to_det[label] = b
    countA = len([l for l in detected_labels if l in groupA])
    countB = len([l for l in detected_labels if l in groupB])
    keep_group = groupA if countA >= countB else groupB
    mirrored_dict = {a: b for a, b in mirrored_pairs}
    mirrored_dict.update({b: a for a, b in mirrored_pairs})
    # Project tracked pitch points and draw ROI triangles on both tactical map and frame
    # Draw ROI triangles on tactical map
    for tri in ROI_REGIONS:
        pts = [pitch_coords.get(str(l)) or pitch_coords.get(int(l)) for l in tri]
        if len(pts) == 3 and all(p is not None for p in pts):
            cv2.polylines(tactical_map, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    # Draw ROI triangles on frame (using detected pitch points)
    detected_pitch_points = {}
    for b in pitch:
        if len(b) > 5:
            label_idx = int(b[5])
            abbr = pitch_model.names[label_idx] if hasattr(pitch_model, 'names') and label_idx in pitch_model.names else str(label_idx)
        else:
            label_idx = None
            abbr = None
        if abbr is None:
            continue
        mapped_abbr = abbr
        if abbr in groupA or abbr in groupB:
            if abbr not in keep_group:
                mapped_abbr = mirrored_dict[abbr]
        label_num = abbreviation_to_number.get(mapped_abbr, mapped_abbr)
        if label_num:
            detected_pitch_points[label_num] = (int(b[0]), int(b[1]))
        if label_num and label_num in pitch_coords:
            x_det, y_det = int(b[0]), int(b[1])
            x_tac, y_tac = pitch_coords[label_num]
            cv2.circle(tactical_map, (int(x_tac), int(y_tac)), 8, (0, 255, 255), -1)
            cv2.putText(tactical_map, f"{abbr} ({label_num})", (int(x_tac)+5, int(y_tac)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    for tri in ROI_REGIONS:
        pts = [detected_pitch_points.get(str(l)) or detected_pitch_points.get(int(l)) for l in tri]
        if all(p is not None for p in pts):
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # Project tracked players onto tactical map using ROI triangles
    for t in online_targets_players:
        x, y, w, h = t.tlwh
        cx, cy = int(x + w/2), int(y + h/2)
        tac_pt = project_point(
            (cx, cy),
            detected_pitch_points,  # use detected pitch points for frame
            pitch_coords,           # use static pitch coords for tactical
            ROI_REGIONS
        )
        if tac_pt is not None:
            tac_cx, tac_cy = tac_pt
            tac_cx = max(0, min(TACTICAL_W - 1, int(tac_cx)))
            tac_cy = max(0, min(TACTICAL_H - 1, int(tac_cy)))
            print(f"[DEBUG][ROI] Player {t.track_id} frame:({cx},{cy}) -> tactical:({tac_cx},{tac_cy})")
            cv2.circle(tactical_map, (tac_cx, tac_cy), 6, (0, 255, 0), -1)
            cv2.putText(tactical_map, f"P{t.track_id}", (tac_cx+8, tac_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Project tracked ball onto tactical map using ROI triangles
    for t in online_targets_ball:
        x, y, w, h = t.tlwh
        cx, cy = int(x + w/2), int(y + h/2)
        tac_pt = project_point(
            (cx, cy),
            detected_pitch_points,
            pitch_coords,
            ROI_REGIONS
        )
        if tac_pt is not None:
            tac_cx, tac_cy = tac_pt
            tac_cx = max(0, min(TACTICAL_W - 1, int(tac_cx)))
            tac_cy = max(0, min(TACTICAL_H - 1, int(tac_cy)))
            print(f"[DEBUG][ROI] Ball frame:({cx},{cy}) -> tactical:({tac_cx},{tac_cy})")
            cv2.circle(tactical_map, (tac_cx, tac_cy), 6, (0, 0, 255), -1)
            cv2.putText(tactical_map, f"Ball", (tac_cx+8, tac_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw tracked players on frame (already updated above)
    for t in online_targets_players:
        x, y, w, h = t.tlwh
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"P{t.track_id}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw tracked ball on frame (already updated above)
    for t in online_targets_ball:
        x, y, w, h = t.tlwh
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.putText(frame, f"B{t.track_id}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ...existing code...

    cv2.imshow("Tracking Test", frame)
    cv2.imshow("Tactical Pitch Projection", tactical_map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
