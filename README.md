# VisionDetect — Smart Object Detection

VisionDetect is a modern, high-fidelity object detection web application powered by **YOLO** (Ultralytics v11) and built using a **React frontend** and a **Node.js Express backend**. It supports uploading images and videos, displaying live bounding boxes synchronized on a video canvas overlay, and exporting full-resolution videos with burned-in annotations.

The interface is styled with a simple, pleasant, light green nature-inspired aesthetic using the **Plus Jakarta Sans** typeface.

---

## Key Features

- **Image Object Detection**: Upload any image (PNG, JPG, WEBP) to detect objects. The client renders the returned annotated image with colored bounding boxes and probability metrics.
- **Video Object Detection**: Upload high-resolution videos (MP4, MOV, WEBM, AVI, etc.) to perform frame-sampled inference. The client extracts video metadata (duration, dimensions) and streams frames to the server.
- **Client-Side Live Canvas Sync**: Plays the uploaded video in an HTML5 video player and matches the current playback frame to YOLO inference results using a high-frequency canvas overlay.
- **Server-Side Video Export**: Re-encodes videos on the server, burning colored bounding boxes and labels into every frame, and streams the finished file back to the browser as an attachment.
- **Predefined YOLO Model Integration**: Integrates a custom model weights file (`rrp32.pt`) loaded on server initialization to eliminate model startup latency.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  React Frontend (Vite Client) - port 5173                   │
│  • Drag-and-drop file upload with metadata extraction       │
│  • Bounding box overlay canvas synced with HTML5 Video RAF  │
│  • Metric totals (Objects found, unique classes, avg conf)  │
└──────────────────────────┬──────────────────────────────────┘
                           │  POST /api/detect
                           │  POST /api/download-annotated-video
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Node.js Express Backend Server - port 5005                 │
│  • Multipart file handling (Multer)                         │
│  • Sequential worker queue to communicate with Python       │
└──────────────────────────┬──────────────────────────────────┘
                           │  stdin / stdout (JSON Lines)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  yolo_worker.py (Persistent Python Worker)                  │
│  • Persistent YOLO model instance loading (rrp32.pt)        │
│  • OpenCV-based frame decoding and video re-encoding        │
│  • Parallel processing of inference & bounding box burning  │
└─────────────────────────────────────────────────────────────┘
```

The system uses a decoupled client-server pattern:
1. **Frontend (`/frontend`)**: A React SPA that captures files and renders analytics.
2. **Backend (`/backend`)**: A Node Express server. To run Python-based YOLO models without spawn delays or heavy C++ bindings, it communicates with a persistent Python process (`yolo_worker.py`) using structured JSON Line streams over `stdin`/`stdout`.

---

## Directory Structure

```
visiondetect/
├── backend/
│   ├── temp_uploads/     # Auto-deleted temporary files
│   ├── server.js         # Express app, endpoint routing & worker queue
│   ├── yolo_worker.py    # Python worker process (YOLO, OpenCV)
│   ├── package.json      # Express, multer, cors, dotenv dependencies
│   └── rrp32.pt          # Custom YOLO model weights file
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DetectionList.jsx  # Metrics and individual label details
│   │   │   └── VideoPlayer.jsx    # Playback synced with bounding box overlay
│   │   │   └── UploadZone.jsx     # Handles drag-drop & metadata parsing
│   │   ├── App.jsx                # App shell, API orchestration, progress bars
│   │   ├── index.css              # Light green organic style system
│   │   └── main.jsx               # React client bootstrap
│   ├── eslint.config.js  # ESLint flat rules configuration
│   ├── package.json      # React, Vite, ESLint dependencies
│   └── vite.config.js    # Vite configurations
└── requirements.txt      # Python system dependencies (ultralytics, opencv, pillow)
```

---

## API Reference

All requests must be made to the Node.js Express server running at `http://localhost:5005`.

### 1. Health Status check
Check if the server and the backend YOLO worker process are alive, model is loaded, and OpenCV libraries are available.
- **Route**: `GET /api/health`
- **Response Format**: `application/json`
- **Output Sample**:
  ```json
  {
    "status": "ok",
    "model": "rrp32.pt",
    "model_loaded": true,
    "load_error": null,
    "classes": {
      "0": "pedestrian",
      "1": "rider",
      "2": "car",
      "3": "bus",
      "4": "truck",
      "5": "bicycle",
      "6": "motorcycle",
      "7": "traffic light",
      "8": "traffic sign",
      "9": "train"
    },
    "cv2_available": true,
    "infer_max_dim": 1280,
    "worker_alive": true,
    "is_ready": true
  }
  ```

---

### 2. Detect Objects
Upload an image or video file to run YOLO inference.
- **Route**: `POST /api/detect`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (Binary File): Image or Video file.
  - `frame_stride` (Form field, default `30`): Frame sampling rate. Process every Nth frame of the video.
  - `max_frames` (Form field, default `10`): Max count of frames to sample and analyze.
  - `conf_threshold` (Form field, default `0.25`): Bounding box confidence cutoff.
  - `iou_threshold` (Form field, default `0.5`): NMS IoU threshold for overlapping boxes.
- **Response Format**: `application/json`
- **Output Sample**:
  ```json
  {
    "detections": [
      {
        "class": "car",
        "confidence": 0.895,
        "bbox": [120.4, 250.1, 480.2, 590.8],
        "frame": 0
      }
    ],
    "raw_detections": [
      {
        "class": "car",
        "confidence": 0.895,
        "bbox": [120.4, 250.1, 480.2, 590.8],
        "frame": 0
      }
    ],
    "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
    "total": 1,
    "video_fps": 30.0,
    "video_width": 1920,
    "video_height": 1080
  }
  ```

---

### 3. Download Annotated Video
Burns bounding boxes directly into every frame of the video based on the raw detections array and returns an MP4 file.
- **Route**: `POST /api/download-annotated-video`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (Binary File): Original raw video file.
  - `detections_json` (Form field): Stringified JSON array of all raw detections (obtained from `/api/detect`).
  - `fps` (Form field, default `30`): Frame rate to encode the output video.
- **Response Format**: `video/mp4` binary stream with header `Content-Disposition: attachment; filename="annotated_<timestamp>.mp4"`.

---

## Getting Started

### Prerequisites
1. **Node.js** (v18.x or v20.x recommended)
2. **Python** (v3.10 or v3.11 recommended)
3. Custom YOLO model weights file (`rrp32.pt`) placed inside the `/backend` folder.

### Installation

1. **Install Python Packages:**
   ```bash
   # From root workspace directory
   pip install -r requirements.txt
   ```

2. **Install Node Backend Dependencies:**
   ```bash
   cd backend
   npm install
   ```

3. **Install React Client Dependencies:**
   ```bash
   cd ../frontend
   npm install
   ```

### Running Locally

1. **Start the Backend Server:**
   ```bash
   cd backend
   npm start
   ```
   *Logs will confirm:*
   ```text
   Starting Python YOLO Worker...
   Express server running on http://localhost:5005
   YOLO Python Worker is READY.
   ```

2. **Start the React Frontend Dev Server:**
   ```bash
   cd ../frontend
   npm run dev
   ```
   *Open your browser to the URL printed in the console (typically `http://localhost:5173`).*

---

## Cleanups & Linting
The frontend comes pre-configured with **ESLint** for code quality audits. Run lints using:
```bash
cd frontend
npm run lint
```
To run a production-ready assets compilation:
```bash
npm run build
```
