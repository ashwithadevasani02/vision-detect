const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const multer = require('multer');

require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const UPLOADS_DIR = path.join(__dirname, 'temp_uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOADS_DIR);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, file.fieldname + '-' + uniqueSuffix + ext);
  }
});
const upload = multer({ storage: storage });

const YOLO_API_URL = process.env.YOLO_API_URL || 'http://localhost:8000';

function safeDelete(filePath) {
  if (filePath && fs.existsSync(filePath)) {
    fs.unlink(filePath, (err) => {
      if (err) console.error(`Error deleting file ${filePath}:`, err);
    });
  }
}

async function sendToFastAPI(endpoint, reqFile, fields = {}) {
  const formData = new FormData();
  
  if (reqFile) {
    const fileBuffer = fs.readFileSync(reqFile.path);
    const fileBlob = new Blob([fileBuffer], { type: reqFile.mimetype });
    formData.append('file', fileBlob, reqFile.originalname);
  }

  for (const [key, value] of Object.entries(fields)) {
    formData.append(key, value);
  }

  const response = await fetch(`${YOLO_API_URL}${endpoint}`, {
    method: 'POST',
    body: formData
  });

  return response;
}

app.use('/static', express.static(path.join(__dirname, 'temp_uploads')));

app.post('/api/detect', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  const filePath = req.file.path;
  const frameStride = req.query.frame_stride || req.body.frame_stride || '30';
  const maxFrames = req.query.max_frames || req.body.max_frames || '10';
  const confThreshold = req.query.conf_threshold || req.body.conf_threshold || '0.25';
  const iouThreshold = req.query.iou_threshold || req.body.iou_threshold || '0.5';

  try {
    const apiResponse = await sendToFastAPI('/detect', req.file, {
      frame_stride: frameStride,
      max_frames: maxFrames,
      conf_threshold: confThreshold,
      iou_threshold: iouThreshold
    });

    safeDelete(filePath);

    if (!apiResponse.ok) {
      const errText = await apiResponse.text();
      let errDetail;
      try {
        errDetail = JSON.parse(errText).detail || errText;
      } catch (e) {
        errDetail = errText;
      }
      return res.status(apiResponse.status).json({ detail: errDetail });
    }

    const data = await apiResponse.json();
    return res.json(data);

  } catch (err) {
    safeDelete(filePath);
    console.error('Inference execution failed:', err);
    return res.status(500).json({ detail: err.message });
  }
});

app.post('/api/download-annotated-video', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  const inputPath = req.file.path;
  const detectionsJson = req.body.detections_json || '[]';
  const fps = req.body.fps || '30.0';

  try {
    const apiResponse = await sendToFastAPI('/export-video', req.file, {
      detections_json: detectionsJson,
      fps: fps
    });

    safeDelete(inputPath);

    if (!apiResponse.ok) {
      const errText = await apiResponse.text();
      let errDetail;
      try {
        errDetail = JSON.parse(errText).detail || errText;
      } catch (e) {
        errDetail = errText;
      }
      return res.status(apiResponse.status).json({ detail: errDetail });
    }

    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Content-Disposition', `attachment; filename="annotated_${Math.floor(Date.now() / 1000)}.mp4"`);

    const { Readable } = require('stream');
    const nodeStream = Readable.fromWeb(apiResponse.body);
    nodeStream.pipe(res);

  } catch (err) {
    safeDelete(inputPath);
    console.error('Video export execution failed:', err);
    return res.status(500).json({ detail: err.message });
  }
});

app.get('/api/health', async (req, res) => {
  try {
    const response = await fetch(`${YOLO_API_URL}/health`);
    if (!response.ok) {
      return res.status(503).json({ error: `YOLO API responded with status ${response.status}` });
    }
    const data = await response.json();
    return res.json({
      ...data,
      worker_alive: true,
      is_ready: data.status === 'ok'
    });
  } catch (err) {
    return res.status(503).json({
      status: 'error',
      message: 'YOLO API service unreachable',
      error: err.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`Express server running on http://localhost:${PORT}`);
  console.log(`Proxying YOLO API calls to ${YOLO_API_URL}`);
});
