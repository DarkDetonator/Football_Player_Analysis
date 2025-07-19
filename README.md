# Football Tracking Project

This project provides a robust football (soccer) tracking pipeline using deep learning models and advanced computer vision techniques. It supports dynamic pitch point detection, player and ball tracking, tactical pitch projection, and ROI overlays for tactical analysis.

## Features
- **Pitch Point Detection:** Detects key pitch points using YOLO-based models, supporting both old and new label formats.
- **Player & Ball Tracking:** Tracks players, referees, and the ball using BYTETracker and YOLO detections.
- **Goalpost Detection:** Detects and maps goalposts for tactical overlays.
- **Dynamic ROI Overlays:** Draws ROI triangles and overlays using dynamically detected pitch points.
- **Tactical Pitch Projection:** Projects tracked objects onto a tactical pitch image for advanced analysis.
- **Overlay Persistence:** Maintains overlays using a cache for robust visualization even with intermittent detections.

## Directory Structure
```
config.py
main.py
requirements.txt
test.py
test_image.py
test_video_pitch_projection.py
assets/
  labels.json
detectors/
  ball_detector.py
  goalpost_detector.py
  pitch_detector.py
  player_detector.py
projection/
  homography_mapper.py
tracking/
  bytetrack_wrapper.py
utils/
  helper.py
visualization/
  visualizer.py
ROI_maping/
  roi_mapper.py
  roi_definitions.py
venv310/
  ... (Python virtual environment)
```

## Setup
1. **Install Python 3.10** (recommended).
2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv310
   .\venv310\Scripts\Activate.ps1
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Configure paths and model weights:**
   - Edit `config.py` and `myconfig.py` to set paths for video, models, and tactical pitch image.
   - Place your YOLO model weights in the appropriate location.

## Usage
- **Run video pitch projection:**
  ```powershell
  python test_video_pitch_projection.py
  ```
- **Run image-based tracking:**
  ```powershell
  python test_image.py
  ```
- **Run tests:**
  ```powershell
  python test.py
  ```

## Customization
- **ROI Regions:**
  - Edit `ROI_maping/roi_definitions.py` to customize ROI triangles and label mappings.
- **Overlay Logic:**
  - Modify `visualization/overlay_utils.py` for custom overlay drawing.
- **Model Weights:**
  - Replace YOLO weights in `detectors/` as needed for improved detection.

## Troubleshooting
- Ensure all paths in `config.py` and `myconfig.py` are correct.
- If overlays or detections are missing, check model weights and label mapping in `ROI_maping/roi_definitions.py`.
- For environment issues, verify your Python version and virtual environment activation.

## Credits
- Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [ByteTrack](https://github.com/ifzhang/ByteTrack).
- Custom tactical pitch mapping and ROI overlay logic by project contributors.

## License
This project is for research and educational purposes. See individual files for license details.
