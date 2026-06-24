# VisionDetect — Smart Object Detection

VisionDetect is a modern, high-fidelity object detection web application powered by **YOLO** (compiled to ONNX format). The project consists of a **React frontend** (Vite), a **Node.js Express Gateway**, and an independent, low-memory **FastAPI YOLO service** utilizing `onnxruntime`.

This memory-optimized setup consumes only **~150MB of RAM** (down from 1.2GB when using PyTorch), making it fully compatible with **Render's Free Tier** (512MB RAM) without hitting Out-Of-Memory (OOM) limits.

> **Note:** The Jupyter notebook used for model training (`model_training.ipynb` or equivalent) is pushed to this repository only for reference. It is not required for running the production application.

---

## System Architecture

```
┌─────────────────────────────────┐
│         React Frontend          │
│          (Port 5173)            │
└────────────────┬────────────────┘
                 │
                 │ HTTP Requests
                 ▼
┌─────────────────────────────────┐
│     Node.js Express Gateway     │
│          (Port 5000)            │
└────────────────┬────────────────┘
                 │
                 │ HTTP API Calls (Multipart Form)
                 ▼
┌─────────────────────────────────┐
│        FastAPI YOLO API         │
│          (Port 8000)            │
│  • Loads model (rrp32.onnx)     │
│  • Runs CPU ONNX Inference      │
└─────────────────────────────────┘
```

The system is split into three parts:
1. **Frontend (`/frontend`)**: A React SPA that captures files and renders metrics, canvas bounding-box overlays, and exports.
2. **Node.js Gateway (`/backend/server.js`)**: An Express app that manages file uploads and acts as a central proxy to direct requests to the FastAPI backend.
3. **FastAPI YOLO API (`/backend/yolo_api.py`)**: A standalone Python service that loads the model weights in ONNX format and performs inference.

---

## Directory Structure

```
visiondetect/
├── backend/
│   ├── temp_uploads/     # Auto-deleted temporary upload files
│   ├── server.js         # Node Express gateway (proxies calls to FastAPI)
│   ├── yolo_api.py       # FastAPI server (onnxruntime predictions)
│   ├── requirements_prod.txt # Production dependencies for Render (no PyTorch)
│   ├── rrp32.onnx        # Exported lightweight ONNX weights file
│   └── package.json      # Express gateway dependencies
├── frontend/
│   ├── src/              # React components & stylesheet
│   ├── package.json      # React dependencies
│   └── vite.config.js    # Vite client configurations
├── requirements.txt      # Local Python dependencies (ultralytics, torch, for training/exporting)
└── README.md             # Project documentation
```

---

## Getting Started

### Prerequisites
1. **Node.js** (v18.x or v20.x recommended)
2. **Python** (v3.10 to v3.13 recommended)

### Installation

1. **Install Node Gateway & React Client Dependencies:**
   ```bash
   # Install Gateway
   cd backend
   npm install

   # Install React Client
   cd ../frontend
   npm install
   ```

2. **Install Python dependencies (Local / Development):**
   ```bash
   # From root workspace directory
   pip install -r requirements.txt
   ```

3. **Install Production Python dependencies (For Render/Production):**
   ```bash
   cd backend
   pip install -r requirements_prod.txt
   ```

---

## Running Locally

To run the application locally, start all three services:

1. **Start the FastAPI Server:**
   ```bash
   cd backend
   uvicorn yolo_api:app --port 8000
   ```
   *The API will run on http://localhost:8000.*

2. **Start the Node.js Express Gateway:**
   ```bash
   cd backend
   npm run dev
   ```
   *The gateway will run on http://localhost:5000 and proxy calls to the FastAPI server.*

3. **Start the React Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```
   *Open http://localhost:5173 in your browser to view the application.*

---

## Production Deployment on Render

### 1. Deploy the FastAPI YOLO service (Web Service)
- **Runtime**: `Python`
- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements_prod.txt`
- **Start Command**: `uvicorn yolo_api:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `MODEL_PATH`: `rrp32.onnx`

### 2. Deploy the Express Gateway (Web Service)
- **Runtime**: `Node`
- **Root Directory**: `backend`
- **Build Command**: `npm install`
- **Start Command**: `npm start`
- **Environment Variables**:
  - `PORT`: `5000` (or leave empty to bind default)
  - `YOLO_API_URL`: `https://your-fastapi-service.onrender.com` (pointing to your deployed FastAPI URL)

### 3. Deploy the Frontend (Static Site)
- Deploy your React build to Render or Vercel, pointing `API_HOST` in the frontend source code to your Node Express gateway URL.
