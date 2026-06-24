import io
import os
import sys
import json
import base64
import time
import traceback
from pathlib import Path
from PIL import Image, ImageDraw

# Set up paths and imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

MODEL_PATH = os.getenv("MODEL_PATH", "rrp32.pt")
model = None
MODEL_LOAD_ERROR = None

# Colors for bounding boxes
COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
]

INFER_MAX_DIM = 1280
VIDEO_EXTS = {"mp4", "mov", "webm", "avi", "mkv", "m4v", "flv", "wmv", "3gp"}

def get_model():
    global model, MODEL_LOAD_ERROR
    if model is not None:
        return model
    if MODEL_LOAD_ERROR is not None:
        raise Exception(f"Model failed to load: {MODEL_LOAD_ERROR}")
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        return model
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        raise Exception(f"Model failed to load: {MODEL_LOAD_ERROR}")

def get_model_names():
    try:
        return get_model().names
    except Exception:
        return {}

def hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def is_video_file(file_path: str) -> bool:
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
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

def handle_detect(req):
    file_path = req["file_path"]
    frame_stride = req.get("frame_stride", 30)
    max_frames = req.get("max_frames", 10)
    conf_threshold = req.get("conf_threshold", 0.25)
    iou_threshold = req.get("iou_threshold", 0.5)

    if not os.path.exists(file_path):
        return {"status": "error", "error": f"File does not exist: {file_path}"}

    # Image
    if not is_video_file(file_path):
        try:
            orig = Image.open(file_path).convert("RGB")
        except Exception as e:
            return {"status": "error", "error": f"Cannot open image: {str(e)}"}
        
        dets = infer_pil(orig, conf_threshold, None)
        annotated = draw_boxes_pil(orig.copy(), dets)
        
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=92)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        
        return {
            "status": "success",
            "data": {
                "detections": dets,
                "raw_detections": dets,
                "annotated_image": f"data:image/jpeg;base64,{img_b64}",
                "total": len(dets),
                "video_fps": None,
                "video_width": orig.width,
                "video_height": orig.height
            }
        }
    
    # Video
    else:
        if not CV2_AVAILABLE:
            return {"status": "error", "error": "opencv-python is not installed on the server."}
        
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return {"status": "error", "error": "OpenCV could not open the video file."}
            
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
                return {"status": "error", "error": "Could not decode any frames from the video."}
            
            detections = nms_merge(all_detections, iou_threshold)
            first_dets = [d for d in all_detections if d.get("frame") == first_fi]
            annotated = draw_boxes_pil(preview_pil.copy(), first_dets)
            
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=92)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            
            return {
                "status": "success",
                "data": {
                    "detections": detections,
                    "raw_detections": all_detections,
                    "annotated_image": f"data:image/jpeg;base64,{img_b64}",
                    "total": len(detections),
                    "video_fps": video_fps,
                    "video_width": int(preview_pil.width),
                    "video_height": int(preview_pil.height),
                }
            }
        except Exception as e:
            return {"status": "error", "error": f"Video processing error: {str(e)}", "traceback": traceback.format_exc()}

def handle_export_video(req):
    file_path = req["file_path"]
    detections = req.get("detections", [])
    fps = req.get("fps", 30.0)
    output_path = req["output_path"]

    if not CV2_AVAILABLE:
        return {"status": "error", "error": "opencv-python is required for video export"}
    if not os.path.exists(file_path):
        return {"status": "error", "error": f"Input video file not found: {file_path}"}

    try:
        frame_map = build_frame_map(detections)
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {"status": "error", "error": "Could not open video file"}

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        if actual_fps <= 0:
            actual_fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, actual_fps, (w, h))
        if not out.isOpened():
            cap.release()
            return {"status": "error", "error": "Failed to create video writer"}

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

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return {"status": "error", "error": "Video encoding produced empty output file"}

        return {"status": "success", "data": {"output_path": output_path}}

    except Exception as e:
        return {"status": "error", "error": f"Video export failed: {str(e)}", "traceback": traceback.format_exc()}

def handle_health(req):
    try:
        m = get_model()
        classes = m.names
        loaded = True
    except Exception as e:
        classes = None
        loaded = False
    
    return {
        "status": "success",
        "data": {
            "status": "ok",
            "model": MODEL_PATH,
            "model_loaded": loaded,
            "load_error": MODEL_LOAD_ERROR,
            "classes": classes,
            "cv2_available": CV2_AVAILABLE,
            "infer_max_dim": INFER_MAX_DIM
        }
    }

def main():
    # Attempt to load model on startup
    try:
        get_model()
    except Exception:
        # Don't crash immediately, allow health check to report the load error
        pass

    # Print READY to stdout to signal Node that startup is complete
    print("READY", flush=True)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            req = json.loads(line.strip())
            action = req.get("action")
            
            if action == "detect":
                res = handle_detect(req)
            elif action == "export_video":
                res = handle_export_video(req)
            elif action == "health":
                res = handle_health(req)
            else:
                res = {"status": "error", "error": f"Unknown action: {action}"}
            
            # Print response as a single JSON line
            print(json.dumps(res), flush=True)

        except KeyboardInterrupt:
            break
        except Exception as e:
            err_res = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
            print(json.dumps(err_res), flush=True)

if __name__ == "__main__":
    main()
