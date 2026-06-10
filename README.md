# VisionDetect

A full-stack object detection web app powered by **YOLO** (Ultralytics). Upload images or videos, run inference with a custom-trained model, and view annotated results in the browser — including live bounding-box overlays on video playback.

## Topics

- [Features](#features)
- [Architecture](#architecture)
- [How Features Are Implemented](#how-features-are-implemented)
  - [Image detection](#1-image-detection)
  - [Video detection](#2-video-detection-frame-sampling)
  - [Live video overlay](#3-live-video-overlay-client-side)
  - [Annotated video export](#4-annotated-video-export)
  - [Configurable detection settings](#5-configurable-detection-settings)
  - [Model loading and environment](#6-model-loading-and-environment)
- [API Reference](#api-reference)
  - [`GET /`](#get-)
  - [`POST /detect`](#post-detect)
  - [`POST /download-annotated-video`](#post-download-annotated-video)
  - [`GET /health`](#get-health)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local development](#local-development)
  - [Docker](#docker)
- [Dependencies](#dependencies)
- [Supported Media Formats](#supported-media-formats)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **Image detection** | Upload PNG, JPG, or WEBP images and get bounding boxes drawn on the result |
| **Video detection** | Sample frames from MP4, MOV, WEBM, AVI, and other common formats |
| **Configurable inference** | Adjust confidence threshold, NMS IoU, frame stride, and max sampled frames |
| **Live video overlay** | Canvas overlay syncs bounding boxes to video playback using nearest-frame matching |
| **Annotated video export** | Server-side re-encoding burns boxes into every frame and returns an MP4 download |
| **Detection summary** | Stats for total objects, unique classes, and average confidence |
| **Health check** | `/health` endpoint reports model path, class names, and dependency status |
| **Docker deployment** | Production-ready container with OpenCV system libraries |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  index.html (frontend)                                      │
│  • Drag-and-drop upload                                     │
│  • Settings panel (stride, frames, conf, IoU)                 │
│  • Canvas video overlay (client-side)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │  POST /detect
                           │  POST /download-annotated-video
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  app.py (FastAPI backend)                                   │
│  • YOLO model inference (Ultralytics)                       │
│  • PIL for image annotation                                 │
│  • OpenCV for video decode/encode                           │
└─────────────────────────────────────────────────────────────┘
```

The frontend is a single-page app served at `/` from `index.html`. The backend loads a YOLO weights file at startup and exposes REST endpoints for detection and video export.

---

## How Features Are Implemented

### 1. Image detection

**Flow:** User uploads an image → `POST /detect` → backend opens it with Pillow → runs YOLO → draws boxes → returns JSON.

**Backend (`app.py`):**

- `is_video_file()` checks MIME type and extension to route image vs video.
- `resize_for_inference()` scales large images so the longest side is at most `INFER_MAX_DIM` (default `1280`) before inference, then `scale_boxes()` maps coordinates back to the original resolution.
- `infer_pil()` calls `model(infer, conf=conf)` and returns a list of `{ class, confidence, bbox, frame }`.
- `draw_boxes_pil()` uses Pillow `ImageDraw` to render colored rectangles and labels.
- The annotated image is JPEG-encoded and returned as a base64 data URL in `annotated_image`.

**Frontend (`index.html`):**

- `setFile()` previews the image via `FileReader`.
- `runDetection()` sends a `multipart/form-data` request with query params for thresholds.
- `showResults()` displays `data.annotated_image` in the result card.

---

### 2. Video detection (frame sampling)

**Flow:** User uploads a video → backend writes it to a temp file → OpenCV reads frames at a configurable stride → YOLO runs on each sampled frame → detections are NMS-merged → first sampled frame is annotated for preview.

**Backend:**

- Requires `opencv-python-headless` (`CV2_AVAILABLE` flag).
- `frame_stride` (default `30`): only every Nth frame is inferred.
- `max_frames` (default `10`): caps how many frames are processed.
- `nms_merge()` applies per-class Non-Maximum Suppression using IoU to deduplicate boxes across sampled frames.
- Response includes:
  - `detections` — NMS-merged list (shown in the UI stats/list)
  - `raw_detections` — all per-frame hits (used for video overlay and export)
  - `video_fps`, `video_width`, `video_height` — metadata for the client overlay

**Frontend:**

- Video thumbnail is extracted from the first frame using a hidden `<video>` + `<canvas>`.
- A metadata badge shows resolution, duration, file size, and MIME type.
- A simulated progress bar runs during upload/inference for UX feedback.

---

### 3. Live video overlay (client-side)

**Implementation:** `syncVideoOverlay()` in `index.html`.

- Plays the original video in a `<video>` element.
- A transparent `<canvas>` sits on top.
- On each animation frame, `currentFrame = floor(currentTime * fps)`.
- `pickFrameDetections()` finds the sampled frame index closest to the current playback frame and draws only those boxes.
- Box colors match the backend palette (`COLORS` array).

This gives a real-time preview without re-encoding the full video on the server.

---

### 4. Annotated video export

**Endpoint:** `POST /download-annotated-video`

**Flow:** Client sends the original video file plus `detections_json` (from `raw_detections`) and `fps` → server decodes every frame → draws boxes with OpenCV → writes MP4 → streams the file back.

**Backend:**

- `build_frame_map()` groups detections by frame index once (avoids per-frame lookup bugs).
- `nearest_frame_dets()` picks the closest sampled frame for each output frame.
- `hex_to_bgr()` converts UI colors to OpenCV BGR order.
- Temp files are cleaned up via FastAPI `BackgroundTasks` after the response is sent.

**Note:** The current frontend UI does not expose a download button for this endpoint, but the API is fully implemented and can be called directly or wired into the UI.

---

### 5. Configurable detection settings

Exposed in the **VIDEO OPTIONS** panel and passed as query parameters to `/detect`:

| Parameter | Query key | Default | Range | Purpose |
|-----------|-----------|---------|-------|---------|
| Frame stride | `frame_stride` | 30 | 1–300 | Skip frames between inferences |
| Max frames | `max_frames` | 10 | 1–120 | Limit sampled frames |
| Confidence | `conf_threshold` | 0.25 | 0.01–1.0 | YOLO confidence cutoff |
| NMS IoU | `iou_threshold` | 0.50 | 0.01–1.0 | IoU threshold for cross-frame dedup |

For images, only `conf_threshold` affects inference; stride and max frames are ignored.

---

### 6. Model loading and environment

At startup, `app.py` loads:

```python
MODEL_PATH = os.getenv("MODEL_PATH", "./rrp32.pt")
model = YOLO(MODEL_PATH)
```

Place your `.pt` weights file in the project root (or set `MODEL_PATH` in a `.env` file). Class names come from the model itself via `model.names`.

Other environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./rrp32.pt` | Path to YOLO weights |
| `INFER_MAX_DIM` | `1280` | Max dimension for inference resize (`0` disables) |
| `PORT` | `8000` (Docker) / `8002` (local `__main__`) | Server port |

---

## API Reference

### `GET /`

Serves the frontend (`index.html`).

### `POST /detect`

Detect objects in an uploaded image or video.

**Body:** `multipart/form-data` with field `file`.

**Query params:** `frame_stride`, `max_frames`, `conf_threshold`, `iou_threshold`

**Response:**

```json
{
  "detections": [{ "class": "person", "confidence": 0.91, "bbox": [10, 20, 100, 200], "frame": 0 }],
  "raw_detections": [...],
  "annotated_image": "data:image/jpeg;base64,...",
  "total": 1,
  "video_fps": 30.0,
  "video_width": 1920,
  "video_height": 1080
}
```

### `POST /download-annotated-video`

Re-encode video with burned-in bounding boxes.

**Body:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Original video |
| `detections_json` | string | JSON array of detections (use `raw_detections` from `/detect`) |
| `fps` | float | Fallback FPS if not readable from file |

**Response:** `video/mp4` stream with `Content-Disposition: attachment`.

### `GET /health`

```json
{
  "status": "ok",
  "model": "./rrp32.pt",
  "classes": { "0": "class_a", "1": "class_b" },
  "cv2_available": true,
  "infer_max_dim": 1280
}
```

---

## Project Structure

```
visondetect/
├── app.py              # FastAPI backend — inference, video processing, API routes
├── index.html          # Frontend SPA — upload UI, overlay, results display
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image for deployment
├── .env                # Local env vars (not committed; create from template below)
└── rrp32.pt            # YOLO weights (not in repo — add your own)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A YOLO `.pt` weights file

### Local development

```bash
# Clone and enter the project
cd visondetect

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Add your model weights
# Copy your .pt file to ./rrp32.pt or set MODEL_PATH in .env

# Optional: create .env
# MODEL_PATH=./rrp32.pt
# INFER_MAX_DIM=1280

# Run the server
python app.py
```

Open [http://localhost:8002](http://localhost:8002) in your browser.

### Docker

```bash
docker build -t visiondetect .
docker run -p 8000:8000 -e PORT=8000 -v /path/to/model.pt:/app/rrp32.pt visiondetect
```

Open [http://localhost:8000](http://localhost:8000).

---

## Dependencies

| Package | Role |
|---------|------|
| `fastapi` | Web framework and API routes |
| `uvicorn` | ASGI server |
| `ultralytics` | YOLO model loading and inference |
| `pillow` | Image open, resize, and PIL-based annotation |
| `opencv-python-headless` | Video decode/encode and OpenCV annotation |
| `python-multipart` | File upload parsing |
| `python-dotenv` | Load `.env` configuration |
| `numpy` | Array operations (Ultralytics dependency) |

---

## Supported Media Formats

**Images:** Any format Pillow can open (PNG, JPG, JPEG, WEBP, etc.)

**Videos:** MP4, MOV, WEBM, AVI, MKV, M4V, FLV, WMV, 3GP (detected by MIME type or extension). For best compatibility, use H.264-encoded MP4.

---

