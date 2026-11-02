# Real-time robot lawnmower detection

Real-time object detection system for recognizing a cat from an
RTSP camera stream. When the configured object is detected with sufficient
confidence, the application sends an HTTP request to a sprinkler controller.

## Core functionality

- Reads live video from an RTSP camera.
- Runs YOLO object detection on each frame.
- Draws bounding boxes and confidence scores on the video.
- Activates an external sprinkler API when a target is detected.
- Uses a cooldown period to prevent repeated API requests.
- Supports webcam and local video sources for demonstrations.
- Provides a dry-run mode that never sends hardware-control requests.
- Handles camera, configuration, and HTTP errors explicitly.

## Architecture

```text
RTSP camera
     |
     v
OpenCV frame capture
     |
     v
YOLO object detection
     |
     v
Confidence filtering
     |
     v
Cooldown protection
     |
     v
Sprinkler HTTP API
```

## Project structure

```text
.
├── main.py             # Application entry point and video loop
├── config.py           # Environment configuration and validation
├── detection.py        # Detection and cooldown logic
├── sprinkler.py        # Sprinkler HTTP client
├── yolov8n.pt          # YOLO model
├── requirements.txt    # Runtime dependencies
└── .env.example        # Configuration template
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with the camera and sprinkler controller settings. Never commit
the real `.env` file because it can contain camera credentials.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `RTSP_USER` | Camera username | required |
| `RTSP_PASSWD` | Camera password | required |
| `CAM_IP` | Camera IP address | required |
| `RTSP_PORT` | RTSP port | required |
| `WEB_SERVER_URL` | Sprinkler controller endpoint | optional |
| `SPRINKLER_TIME` | Sprinkler activation time in seconds | `5` |
| `CAT_API_COOLDOWN_SECONDS` | Minimum time between activations | `60` |
| `CONFIDENCE_THRESHOLD` | Minimum detection confidence | `0.75` |
| `TARGET_CLASSES` | Comma-separated target classes | `robot_lawnmower` |

## Running

### RTSP camera

```bash
python main.py --model path/to/model.pt
```

When `--source` is omitted, the RTSP URL is built from `.env`.

### Webcam or video file

```bash
python main.py --source 0 --dry-run
python main.py --source demo.mp4 --dry-run
```

The `--dry-run` option logs sprinkler activations without sending an HTTP
request. Press `q` in the video window to stop the application.

On a Wayland desktop where the OpenCV Qt window cannot find the Wayland
plugin, use the X11 compatibility backend:

```bash
QT_QPA_PLATFORM=xcb python main.py --source 0 --dry-run
```
