import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
from skimage import color as skimage_color
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.vq import kmeans, vq
try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    print("Warning: 'moviepy' not installed. Video saving will use OpenCV. Run 'pip install moviepy' and ensure ffmpeg is installed.")
    ImageSequenceClip = None
from tqdm import tqdm
import yaml
import json
import time
import os

# Configuration class to manage parameters
class Config:
    def __init__(self, config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {config_path} not found.")
        self.video_path = cfg['video_path']
        self.tactical_map_path = cfg['tactical_map_path']
        self.keypoints_json = cfg['keypoints_json']
        self.player_model_path = cfg['player_model_path']
        self.keypoints_model_path = cfg['keypoints_model_path']
        self.player_conf_thresh = cfg.get('player_conf_thresh', 0.60)
        self.keypoints_conf_thresh = cfg.get('keypoints_conf_thresh', 0.70)
        self.keypoints_displacement_tol = cfg.get('keypoints_displacement_tol', 10)
        self.ball_track_dist_thresh = cfg.get('ball_track_dist_thresh', 100)
        self.nbr_frames_no_ball_thresh = cfg.get('nbr_frames_no_ball_thresh', 30)
        self.max_track_length = cfg.get('max_track_length', 35)
        self.team_colors = cfg['team_colors']
        self.output_video_path = cfg.get('output_video_path', 'output_video.mp4')

# Load configurations
config_path = "config.yaml"
try:
    config = Config(config_path)
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)

# Load tactical map and keypoints
try:
    tac_map = cv2.imread(config.tactical_map_path)
    if tac_map is None:
        raise ValueError(f"Failed to load tactical map from {config.tactical_map_path}")
    with open(config.keypoints_json, 'r') as f:
        keypoints_map_pos = json.load(f)
except Exception as e:
    print(f"Error loading resources: {e}")
    exit(1)

# Initialize YOLO models
try:
    model_players = YOLO(config.player_model_path)
    model_keypoints = YOLO(config.keypoints_model_path)
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    exit(1)

# Initialize tracking data
tracking_data = {'frame': [], 'player_id': [], 'team': [], 'x': [], 'y': [], 'ball_x': [], 'ball_y': []}
ball_track_history = {'src': [], 'dst': []}
nbr_frames_no_ball = 0

# Load class names for keypoints and labels
try:
    with open(r"D:\AI platoform fo rvisual cognative impared\New folder\Football-Analytics-with-Deep-Learning-and-Computer-Vision\config pitch dataset.yaml", 'r') as file:
        pitch_yaml = yaml.safe_load(file)
        if not pitch_yaml or 'names' not in pitch_yaml:
            raise ValueError("Invalid or empty 'names' in config pitch dataset.yaml")
        classes_names_dic = {int(k): v for k, v in pitch_yaml['names'].items()}
    with open(r"D:\AI platoform fo rvisual cognative impared\New folder\Football-Analytics-with-Deep-Learning-and-Computer-Vision\config players dataset.yaml", 'r') as file:
        players_yaml = yaml.safe_load(file)
        if not players_yaml or 'names' not in players_yaml:
            raise ValueError("Invalid or empty 'names' in config players dataset.yaml")
        labels_dic = {int(k): v for k, v in players_yaml['names'].items()}
except Exception as e:
    print(f"Error loading YAML configs: {e}")
    exit(1)

# Convert team colors to L*a*b* space
colors_list = sum(config.team_colors.values(), [])
color_list_lab = [skimage_color.rgb2lab([i/255 for i in c]) for c in colors_list]

def detect_objects(frame, model_players, model_keypoints, player_conf_thresh, keypoints_conf_thresh):
    """Run YOLO inference for players and keypoints."""
    try:
        results_players = model_players(frame, conf=player_conf_thresh)
        results_keypoints = model_keypoints(frame, conf=keypoints_conf_thresh)
        return results_players, results_keypoints
    except Exception as e:
        print(f"Error during YOLO inference: {e}")
        return None, None

def get_team_color(bbox, frame, colors_list):
    """Predict team based on player bounding box color."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "Unknown"
        avg_color = np.mean(roi, axis=(0, 1))
        avg_color_lab = skimage_color.rgb2lab(avg_color / 255.0)
        distances = [np.linalg.norm(avg_color_lab - c) for c in color_list_lab]
        team_idx = np.argmin(distances)
        return list(config.team_colors.keys())[team_idx // 2]
    except Exception as e:
        print(f"Error in team color detection: {e}")
        return "Unknown"

def compute_homography(keypoints_detected, keypoints_map_pos, displacement_tol):
    """Compute homography matrix for perspective transformation."""
    try:
        src_pts = []
        dst_pts = []
        for det in keypoints_detected:
            label = int(det[-1])
            if label in classes_names_dic:
                keypoint_name = classes_names_dic[label]
                if keypoint_name in keypoints_map_pos:
                    src_pts.append(det[:2])
                    dst_pts.append([keypoints_map_pos[keypoint_name]['x'], keypoints_map_pos[keypoint_name]['y']])
        if len(src_pts) >= 4:
            src_pts = np.array(src_pts, dtype=np.float32)
            dst_pts = np.array(dst_pts, dtype=np.float32)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            return H
        return None
    except Exception as e:
        print(f"Error computing homography: {e}")
        return None

def cluster_players(positions, n_clusters=3):
    """Cluster players to identify formations."""
    try:
        if len(positions) < n_clusters:
            return [0] * len(positions)
        centroids, _ = kmeans(positions, n_clusters)
        cluster_ids, _ = vq(positions, centroids)
        return cluster_ids
    except Exception as e:
        print(f"Error in clustering: {e}")
        return [0] * len(positions)

def generate_heatmap(positions, tac_map_shape):
    """Generate heatmap of player positions."""
    try:
        heatmap, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1], 
            bins=(tac_map_shape[1] // 10, tac_map_shape[0] // 10),
            range=[[0, tac_map_shape[1]], [0, tac_map_shape[0]]]
        )
        return heatmap
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return np.zeros((tac_map_shape[0] // 10, tac_map_shape[1] // 10))

def process_frame(frame, frame_nbr, prev_H, prev_frame_time):
    """Process a single video frame."""
    global ball_track_history, nbr_frames_no_ball
    tac_map_copy = tac_map.copy()
    
    # Detect objects
    results_players, results_keypoints = detect_objects(frame, model_players, model_keypoints, 
                                                       config.player_conf_thresh, config.keypoints_conf_thresh)
    if results_players is None or results_keypoints is None:
        return frame, prev_H, prev_frame_time
    
    # Extract bounding boxes and keypoints
    try:
        bboxes_p = results_players[0].boxes.xyxy.cpu().numpy()
        classes_p = results_players[0].boxes.cls.cpu().numpy()
        keypoints_detected = results_keypoints[0].boxes.xyxy.cpu().numpy()
    except Exception as e:
        print(f"Error extracting detection results: {e}")
        return frame, prev_H, prev_frame_time
    
    # Compute homography
    H = compute_homography(keypoints_detected, keypoints_map_pos, config.keypoints_displacement_tol)
    if H is None and prev_H is not None:
        H = prev_H
    
    # Process players
    player_positions = []
    for bbox, cls in zip(bboxes_p, classes_p):
        if cls == 0:  # Player
            team = get_team_color(bbox, frame, config.team_colors)
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            if H is not None:
                transformed = cv2.perspectiveTransform(np.array([[center]], dtype=np.float32), H)[0][0]
                player_positions.append(transformed)
                cv2.circle(tac_map_copy, (int(transformed[0]), int(transformed[1])), 5, 
                           config.team_colors[team][0], -1)
                tracking_data['frame'].append(frame_nbr)
                tracking_data['player_id'].append(len(tracking_data['player_id']))
                tracking_data['team'].append(team)
                tracking_data['x'].append(transformed[0])
                tracking_data['y'].append(transformed[1])
    
    # Cluster players for formation analysis
    if player_positions:
        player_positions = np.array(player_positions)
        cluster_ids = cluster_players(player_positions, n_clusters=3)
        for pos, cid in zip(player_positions, cluster_ids):
            cv2.putText(tac_map_copy, f"C{cid}", (int(pos[0]), int(pos[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Process ball
    ball_detected = False
    for bbox, cls in zip(bboxes_p, classes_p):
        if cls == 2:  # Ball
            ball_detected = True
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            if H is not None:
                transformed = cv2.perspectiveTransform(np.array([[center]], dtype=np.float32), H)[0][0]
                ball_track_history['src'].append(center)
                ball_track_history['dst'].append(transformed)
                if len(ball_track_history['dst']) > 1:
                    cv2.line(tac_map_copy, 
                             (int(ball_track_history['dst'][-2][0]), int(ball_track_history['dst'][-2][1])),
                             (int(transformed[0]), int(transformed[1])), (0, 255, 0), 2)
                tracking_data['ball_x'].append(transformed[0])
                tracking_data['ball_y'].append(transformed[1])
    
    # Reset ball track if not detected for too long
    if not ball_detected:
        nbr_frames_no_ball += 1
        if nbr_frames_no_ball > config.nbr_frames_no_ball_thresh:
            ball_track_history['src'] = []
            ball_track_history['dst'] = []
    else:
        nbr_frames_no_ball = 0
    
    # Generate heatmap
    if player_positions:
        heatmap = generate_heatmap(player_positions, tac_map.shape)
        heatmap = cv2.resize(heatmap, (tac_map.shape[1], tac_map.shape[0]))
        heatmap = (heatmap / (heatmap.max() + 1e-10) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        tac_map_copy = cv2.addWeighted(tac_map_copy, 0.7, heatmap, 0.3, 0)
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    
    # Combine frames
    annotated_frame = results_players[0].plot()
    border_color = [255, 255, 255]
    annotated_frame = cv2.copyMakeBorder(annotated_frame, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
    tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
    tac_map_copy = cv2.resize(tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0]))
    final_img = cv2.hconcat((annotated_frame, tac_map_copy))
    
    # Add annotations
    cv2.putText(final_img, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(final_img, "Tactical Map", (1370, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    return final_img, H, new_frame_time

def main():
    """Main processing loop."""
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {config.video_path}")
        return
    
    # Initialize video writer
    frames = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.output_video_path, fourcc, 30.0, 
                          (tac_map.shape[1] + 1280, tac_map.shape[0]))
    
    prev_H = None
    prev_frame_time = time.time()
    frame_nbr = 0
    
    try:
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing frames") as pbar:
            while cap.isOpened():
                frame_nbr += 1
                success, frame = cap.read()
                if not success:
                    break
                
                final_img, prev_H, prev_frame_time = process_frame(frame, frame_nbr, prev_H, prev_frame_time)
                frames.append(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
                out.write(final_img)
                
                # Display using Matplotlib for compatibility
                plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.001)
                
                # Handle keyboard input
                try:
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    if key == ord('p'):
                        cv2.waitKey(-1)
                except Exception as e:
                    print(f"Error handling keyboard input: {e}. Using matplotlib pause.")
                    plt.pause(0.1)
                
                pbar.update(1)
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        cap.release()
        out.release()
        plt.close('all')
        # Save tracking data
        pd.DataFrame(tracking_data).to_csv('tracking_data.csv', index=False)
        # Save video using moviepy if available
        if ImageSequenceClip is not None:
            try:
                clip = ImageSequenceClip(frames, fps=30)
                clip.write_videofile(config.output_video_path, codec='libx264')
                print(f"Video saved to {config.output_video_path} using moviepy")
            except Exception as e:
                print(f"Error saving video with moviepy: {e}. Video saved using OpenCV instead.")
        else:
            print(f"Video saved to {config.output_video_path} using OpenCV.")

if __name__ == "__main__":
    main()