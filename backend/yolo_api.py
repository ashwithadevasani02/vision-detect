import os
import io
import json
import base64
import shutil
import tempfile
import traceback
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

app = FastAPI(title="YOLO ONNX Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "rrp32.onnx")
session = None
CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bicycle",
    6: "motorcycle",
    7: "traffic light",
    8: "traffic sign",
    9: "train"
}

COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
]

VIDEO_EXTS = {"mp4", "mov", "webm", "avi", "mkv", "m4v", "flv", "wmv", "3gp"}

def get_session():
    global session, CLASS_NAMES
    if session is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"ONNX model file not found at '{MODEL_PATH}'. Please run export_onnx.py first.")
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        try:
            meta = session.get_modelmeta()
            custom_meta = meta.custom_metadata_map
            if 'names' in custom_meta:
                import ast
                CLASS_NAMES = ast.literal_eval(custom_meta['names'])
        except Exception:
            pass
    return session

def hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def is_video_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in VIDEO_EXTS

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

def infer_onnx(pil_img: Image.Image, conf_threshold: float, frame_idx=None) -> list:
    orig_w, orig_h = pil_img.size
    
    img_resized = pil_img.resize((640, 640), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    
    sess = get_session()
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: img_np})
    
    pred = outputs[0]
    detections = []
    
    if len(pred.shape) == 3 and pred.shape[2] == 6:
        boxes = pred[0]
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            if score < conf_threshold:
                continue
            
            x1 = float(x1) * (orig_w / 640.0)
            y1 = float(y1) * (orig_h / 640.0)
            x2 = float(x2) * (orig_w / 640.0)
            y2 = float(y2) * (orig_h / 640.0)
            
            cls_name = CLASS_NAMES.get(int(class_id), f"class_{int(class_id)}")
            detections.append({
                "class": cls_name,
                "confidence": round(float(score), 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "frame": frame_idx
            })
    else:
        if len(pred.shape) == 3:
            pred = pred[0]
        pred = pred.T
        
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask = confidences >= conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        for box, conf, cid in zip(boxes, confidences, class_ids):
            cx, cy, w, h = box
            x1 = (cx - w / 2.0) * (orig_w / 640.0)
            y1 = (cy - h / 2.0) * (orig_h / 640.0)
            x2 = (cx + w / 2.0) * (orig_w / 640.0)
            y2 = (cy + h / 2.0) * (orig_h / 640.0)
            
            cls_name = CLASS_NAMES.get(int(cid), f"class_{int(cid)}")
            detections.append({
                "class": cls_name,
                "confidence": round(float(conf), 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "frame": frame_idx
            })
            
    return detections

def build_frame_map(detections: list) -> dict:
    from collections import defaultdict
    frame_map = defaultdict(list)
    for det in detections:
        frame_map[det.get("frame", 0)].append(det)
    return dict(frame_map)

def nearest_frame_dets(frame_idx: int, frame_map: dict) -> list:
    if not frame_map:
        return []
    closest = min(frame_map.keys(), key=lambda f: abs(f - frame_idx))
    return frame_map[closest]

def remove_temp_file(path: str):
    if os.path.exists(path):
        try:
            os.unlink(path)
        except Exception:
            pass

@app.get("/health")
def health_check():
    try:
        get_session()
        loaded = True
        err = None
    except Exception as e:
        loaded = False
        err = str(e)
    
    return {
        "status": "ok" if loaded else "error",
        "model": MODEL_PATH,
        "model_loaded": loaded,
        "load_error": err,
        "classes": CLASS_NAMES if loaded else None,
        "cv2_available": CV2_AVAILABLE
    }

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    frame_stride: int = Form(30),
    max_frames: int = Form(10),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.5)
):
    try:
        get_session()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {str(e)}")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if not is_video_file(file.filename):
            try:
                orig = Image.open(tmp_path).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cannot open image: {str(e)}")
            
            dets = infer_onnx(orig, conf_threshold, None)
            merged_dets = nms_merge(dets, iou_threshold)
            
            annotated = draw_boxes_pil(orig.copy(), merged_dets)
            
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=92)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            
            return {
                "detections": merged_dets,
                "raw_detections": merged_dets,
                "annotated_image": f"data:image/jpeg;base64,{img_b64}",
                "total": len(merged_dets),
                "video_fps": None,
                "video_width": orig.width,
                "video_height": orig.height
            }
        else:
            if not CV2_AVAILABLE:
                raise HTTPException(status_code=500, detail="opencv-python is not installed on the server.")
            
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="OpenCV could not open the video file.")
            
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
                    frame_dets = infer_onnx(pil, conf_threshold, frame_idx)
                    all_detections.extend(frame_dets)
                    sampled_frames += 1
                frame_idx += 1
            
            cap.release()
            
            if preview_pil is None:
                raise HTTPException(status_code=400, detail="Could not decode any frames from the video.")
            
            detections = nms_merge(all_detections, iou_threshold)
            first_dets = [d for d in all_detections if d.get("frame") == first_fi]
            annotated = draw_boxes_pil(preview_pil.copy(), first_dets)
            
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=92)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            
            return {
                "detections": detections,
                "raw_detections": all_detections,
                "annotated_image": f"data:image/jpeg;base64,{img_b64}",
                "total": len(detections),
                "video_fps": video_fps,
                "video_width": int(preview_pil.width),
                "video_height": int(preview_pil.height),
            }
            
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        remove_temp_file(tmp_path)

@app.post("/export-video")
async def export_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detections_json: str = Form("[]"),
    fps: float = Form(30.0)
):
    if not CV2_AVAILABLE:
        raise HTTPException(status_code=500, detail="opencv-python is required for video export")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        input_path = tmp_in.name

    output_fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(output_fd)

    try:
        detections = json.loads(detections_json)
        frame_map = build_frame_map(detections)
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        if actual_fps <= 0:
            actual_fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, actual_fps, (w, h))
        if not out.isOpened():
            cap.release()
            raise HTTPException(status_code=500, detail="Failed to create video writer")

        frame_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break

            frame_dets = nearest_frame_dets(frame_idx, frame_map)

            for i, det in enumerate(frame_dets):
                bbox = det.get("bbox", [0, 0, w, h])
                x1, y1, x2, y2 = map(int, bbox)
                color = hex_to_bgr(COLORS[i % len(COLORS)])
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

        background_tasks.add_task(remove_temp_file, output_path)

        return FileResponse(
            path=output_path,
            filename=f"annotated_{int(os.path.getmtime(output_path))}.mp4",
            media_type="video/mp4"
        )

    except Exception as e:
        print(traceback.format_exc())
        remove_temp_file(output_path)
        raise HTTPException(status_code=500, detail=f"Video export failed: {str(e)}")
    finally:
        remove_temp_file(input_path)
