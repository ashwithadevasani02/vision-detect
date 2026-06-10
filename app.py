from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
from pathlib import Path
import io, base64, os, uvicorn, tempfile, traceback, json, time

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

app = FastAPI(title="YOLO Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "./rrp32.pt")
model = None
MODEL_LOAD_ERROR = None


def get_model() -> YOLO:
    global model, MODEL_LOAD_ERROR
    if model is not None:
        return model
    if MODEL_LOAD_ERROR is not None:
        raise HTTPException(503, f"Model failed to load: {MODEL_LOAD_ERROR}")
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        raise HTTPException(503, f"Model failed to load: {MODEL_LOAD_ERROR}")


def get_model_names() -> dict:
    return get_model().names


FRONTEND_FILE = Path(__file__).with_name("index.html")
if not FRONTEND_FILE.exists():
    FRONTEND_FILE = Path(__file__).with_name("index.html")

COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
]

INFER_MAX_DIM = int(os.getenv("INFER_MAX_DIM", "1280"))
VIDEO_EXTS = {"mp4", "mov", "webm", "avi", "mkv", "m4v", "flv", "wmv", "3gp"}


# ── Helpers ──────────────────────────────────────────────────────

def hex_to_bgr(hex_color: str) -> tuple:
    """Convert #RRGGBB to OpenCV BGR tuple (channel order is reversed vs RGB)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def is_video_file(content_type: str, filename: str) -> bool:
    if content_type and content_type.startswith("video/"):
        return True
    ext = (filename or "").rsplit(".", 1)[-1].lower()
    return ext in VIDEO_EXTS


def resize_for_inference(image: Image.Image) -> Image.Image:
    if INFER_MAX_DIM <= 0:
        return image
    w, h = image.size
    if max(w, h) <= INFER_MAX_DIM:
        return image
    scale = INFER_MAX_DIM / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def scale_boxes(boxes_xyxy, infer_size, orig_size):
    iw, ih = infer_size
    ow, oh = orig_size
    sx, sy = ow / iw, oh / ih
    return [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes_xyxy]


def draw_boxes_pil(image: Image.Image, detections: list) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = COLORS[i % len(COLORS)]
        label = f"{det['class']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1 - 22, x1 + len(label) * 8 + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - 19), label, fill="white")
    return image


def infer_pil(pil_img: Image.Image, conf: float, frame_idx) -> list:
    infer = resize_for_inference(pil_img)
    current_model = get_model()
    results = current_model(infer, conf=conf, verbose=False)
    scaled = scale_boxes(
        [box.xyxy[0].tolist() for box in results[0].boxes],
        infer.size, pil_img.size,
    )
    return [{
        "class":      current_model.names[int(box.cls)],
        "confidence": round(float(box.conf), 3),
        "bbox":       [round(v, 1) for v in scaled[idx]],
        "frame":      frame_idx,
    } for idx, box in enumerate(results[0].boxes)]


def nms_merge(detections: list, iou_thr: float) -> list:
    if not detections:
        return []
    from collections import defaultdict
    by_class = defaultdict(list)
    for d in detections:
        by_class[d["class"]].append(d)
    kept = []
    for dets in by_class.values():
        dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)
        sup = [False] * len(dets)
        for i in range(len(dets)):
            if sup[i]:
                continue
            kept.append(dets[i])
            x1i, y1i, x2i, y2i = dets[i]["bbox"]
            for j in range(i + 1, len(dets)):
                if sup[j]:
                    continue
                x1j, y1j, x2j, y2j = dets[j]["bbox"]
                ix = max(0, min(x2i, x2j) - max(x1i, x1j))
                iy = max(0, min(y2i, y2j) - max(y1i, y1j))
                inter = ix * iy
                union = (x2i - x1i) * (y2i - y1i) + (x2j - x1j) * (y2j - y1j) - inter
                if union > 0 and inter / union > iou_thr:
                    sup[j] = True
    return kept


def build_frame_map(detections: list) -> dict:
    """Group raw per-frame detections by their frame index."""
    from collections import defaultdict
    frame_map = defaultdict(list)
    for det in detections:
        frame_map[det.get("frame", 0)].append(det)
    return dict(frame_map)


def nearest_frame_dets(frame_idx: int, frame_map: dict) -> list:
    """Return detections from the sampled frame closest to frame_idx."""
    if not frame_map:
        return []
    closest = min(frame_map.keys(), key=lambda f: abs(f - frame_idx))
    return frame_map[closest]


def cleanup_temp_file(file_path: str):
    try:
        time.sleep(2)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Could not cleanup {file_path}: {e}")


# ── Routes ───────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_FILE))


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    frame_stride:   int   = Query(default=30,   ge=1,    le=300),
    max_frames:     int   = Query(default=10,   ge=1,    le=120),
    conf_threshold: float = Query(default=0.25, ge=0.01, le=1.0),
    iou_threshold:  float = Query(default=0.5,  ge=0.01, le=1.0),
):
    content_type = (file.content_type or "").lower()
    filename = file.filename or ""

    video_fps = None
    preview_pil = None
    detections = []
    all_detections = []
    annotated = None

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read uploaded file: {e}")

    if len(contents) == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    # ── Image ────────────────────────────────────────────────────
    if not is_video_file(content_type, filename):
        try:
            orig = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"Cannot open image: {e}")
        dets = infer_pil(orig, conf_threshold, None)
        annotated = draw_boxes_pil(orig.copy(), dets)
        detections = dets
        all_detections = dets
        preview_pil = orig

    # ── Video ────────────────────────────────────────────────────
    else:
        if not CV2_AVAILABLE:
            raise HTTPException(500, "opencv-python is not installed. Run: pip install opencv-python")

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp4"
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise HTTPException(400, "OpenCV could not open the video. Try re-encoding to H.264 MP4.")

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if video_fps <= 0:
                video_fps = 30.0

            all_detections = []
            preview_pil = None
            first_fi = None

            frame_idx = 0
            sampled_frames = 0
            while sampled_frames < max_frames:
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break
                if frame_idx % frame_stride == 0:
                    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    if preview_pil is None:
                        preview_pil = pil.copy()
                        first_fi = frame_idx
                    frame_dets = infer_pil(pil, conf_threshold, frame_idx)
                    all_detections.extend(frame_dets)
                    sampled_frames += 1
                frame_idx += 1

            cap.release()

            if preview_pil is None:
                raise HTTPException(500, "Could not decode any frames from the video.")

            detections = nms_merge(all_detections, iou_threshold)
            first_dets = [d for d in all_detections if d.get("frame") == first_fi]
            annotated = draw_boxes_pil(preview_pil.copy(), first_dets)

        except HTTPException:
            raise
        except Exception:
            raise HTTPException(500, f"Video processing failed: {traceback.format_exc()}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ── Encode & respond ─────────────────────────────────────────
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "detections":      detections,       # NMS-merged, shown in UI list
        "raw_detections":  all_detections,   # full per-frame data for video export
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        "total":           len(detections),
        "video_fps":       video_fps,
        "video_width":     int(preview_pil.width),
        "video_height":    int(preview_pil.height),
    }


@app.post("/download-annotated-video")
async def download_annotated_video(
    file:             UploadFile = File(...),
    # FIX: Form(...) is required — plain type hints are ignored in multipart requests
    detections_json:  str        = Form(default="[]"),
    fps:              float      = Form(default=30.0),
    background_tasks: BackgroundTasks = None,
):
    """Re-encode video with bounding boxes burned into every frame."""
    if not CV2_AVAILABLE:
        raise HTTPException(500, "opencv-python is required for video export")

    try:
        detections = json.loads(detections_json) if detections_json else []
    except json.JSONDecodeError:
        detections = []

    # Build frame lookup ONCE — avoids the per-frame scoping bug in the original
    frame_map = build_frame_map(detections)

    tmp_input = None
    tmp_output = None

    try:
        contents = await file.read()
        ext = (file.filename or "video").rsplit(".", 1)[-1].lower() \
              if "." in (file.filename or "") else "mp4"

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(contents)
            tmp_input = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_output = tmp.name

        cap = cv2.VideoCapture(tmp_input)
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        if actual_fps <= 0:
            actual_fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_output, fourcc, actual_fps, (w, h))
        if not out.isOpened():
            cap.release()
            raise HTTPException(500, "Failed to create video writer")

        frame_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break

            # Nearest-frame lookup using the pre-built map
            frame_dets = nearest_frame_dets(frame_idx, frame_map)

            for i, det in enumerate(frame_dets):
                x1, y1, x2, y2 = map(int, det.get("bbox", [0, 0, w, h]))
                color = hex_to_bgr(COLORS[i % len(COLORS)])  # FIX: BGR not RGB
                label = f"{det.get('class', '?')} {det.get('confidence', 0):.2f}"

                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(bgr, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
                cv2.putText(bgr, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            out.write(bgr)
            frame_idx += 1

        cap.release()
        out.release()

        if not os.path.exists(tmp_output) or os.path.getsize(tmp_output) == 0:
            raise HTTPException(500, "Video encoding produced no output")

        with open(tmp_output, "rb") as f:
            video_bytes = f.read()

        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, tmp_input)
            background_tasks.add_task(cleanup_temp_file, tmp_output)

        filename_out = f"annotated_{int(time.time())}.mp4"
        return StreamingResponse(
            io.BytesIO(video_bytes),
            media_type="video/mp4",
            headers={"Content-Disposition": f'attachment; filename="{filename_out}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Video encoding error: {traceback.format_exc()}")
        for p in [tmp_input, tmp_output]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        raise HTTPException(500, f"Video encoding failed: {str(e)}")


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model":         MODEL_PATH,
        "model_loaded":  model is not None,
        "load_error":    MODEL_LOAD_ERROR,
        "classes":       get_model_names() if model is not None else None,
        "cv2_available": CV2_AVAILABLE,
        "infer_max_dim": INFER_MAX_DIM,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=False)