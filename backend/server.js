const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const readline = require('readline');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5005;

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Setup temporary upload directory
const UPLOADS_DIR = path.join(__dirname, 'temp_uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Multer storage config
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

// ── YOLO Python Worker Manager ──────────────────────────────────────────

let worker = null;
let rl = null;
let pendingResolve = null;
let pendingReject = null;
const queue = [];
let isReady = false;

function startWorker() {
  console.log('Starting Python YOLO Worker...');

  // Spawn Python worker inside backend directory
  worker = spawn('python', ['yolo_worker.py'], {
    cwd: __dirname
  });

  isReady = false;

  rl = readline.createInterface({
    input: worker.stdout,
    terminal: false
  });

  rl.on('line', (line) => {
    if (line.trim() === 'READY') {
      console.log('YOLO Python Worker is READY.');
      isReady = true;
      processQueue();
      return;
    }

    try {
      const response = JSON.parse(line);
      if (pendingResolve) {
        pendingResolve(response);
      }
    } catch (err) {
      console.error('Failed to parse stdout line as JSON:', line);
      if (pendingReject) {
        pendingReject(err);
      }
    }

    pendingResolve = null;
    pendingReject = null;
    processQueue();
  });

  worker.stderr.on('data', (data) => {
    console.error(`Python Worker STDERR: ${data.toString()}`);
  });

  worker.on('exit', (code) => {
    console.warn(`Python Worker exited with code ${code}.`);
    isReady = false;

    // Reject any pending promise
    if (pendingReject) {
      pendingReject(new Error('YOLO Python Worker crashed during execution.'));
      pendingResolve = null;
      pendingReject = null;
    }

    // Automatically restart after 1 second
    console.log('Restarting YOLO Python Worker in 1 second...');
    setTimeout(startWorker, 1000);
  });
}

function processQueue() {
  if (queue.length === 0 || pendingResolve !== null || !isReady) {
    return;
  }

  const { command, resolve, reject } = queue.shift();
  pendingResolve = resolve;
  pendingReject = reject;

  try {
    worker.stdin.write(JSON.stringify(command) + '\n');
  } catch (err) {
    reject(err);
    pendingResolve = null;
    pendingReject = null;
    processQueue();
  }
}

function runCommand(command) {
  return new Promise((resolve, reject) => {
    queue.push({ command, resolve, reject });
    processQueue();
  });
}

// Start the worker on server load
startWorker();

// Helper to cleanup file safely
function safeDelete(filePath) {
  if (filePath && fs.existsSync(filePath)) {
    fs.unlink(filePath, (err) => {
      if (err) console.error(`Error deleting file ${filePath}:`, err);
    });
  }
}

// ── API Routes ──────────────────────────────────────────────────────────

// Serve frontend build static files if needed (placeholder, handled by client dev server normally)
app.use('/static', express.static(path.join(__dirname, 'temp_uploads')));

// 1. POST /api/detect
app.post('/api/detect', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  const filePath = req.file.path;

  // Parse query parameters/form fields
  const frameStride = parseInt(req.query.frame_stride || req.body.frame_stride || 30);
  const maxFrames = parseInt(req.query.max_frames || req.body.max_frames || 10);
  const confThreshold = parseFloat(req.query.conf_threshold || req.body.conf_threshold || 0.25);
  const iouThreshold = parseFloat(req.query.iou_threshold || req.body.iou_threshold || 0.5);

  try {
    const result = await runCommand({
      action: 'detect',
      file_path: filePath,
      frame_stride: frameStride,
      max_frames: maxFrames,
      conf_threshold: confThreshold,
      iou_threshold: iouThreshold
    });

    // Cleanup input upload file
    safeDelete(filePath);

    if (result.status === 'error') {
      return res.status(500).json({ detail: result.error });
    }

    return res.json(result.data);

  } catch (err) {
    safeDelete(filePath);
    console.error('Inference execution failed:', err);
    return res.status(500).json({ detail: err.message });
  }
});

// 2. POST /api/download-annotated-video
app.post('/api/download-annotated-video', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  const inputPath = req.file.path;
  const outputPath = path.join(UPLOADS_DIR, `annotated-${Date.now()}.mp4`);

  const detectionsJson = req.body.detections_json || '[]';
  const fps = parseFloat(req.body.fps || 30.0);

  let detections = [];
  try {
    detections = JSON.parse(detectionsJson);
  } catch (e) {
    console.warn('Failed to parse detections_json, using empty array.');
  }

  try {
    const result = await runCommand({
      action: 'export_video',
      file_path: inputPath,
      detections: detections,
      fps: fps,
      output_path: outputPath
    });

    // Delete the input uploaded file immediately
    safeDelete(inputPath);

    if (result.status === 'error') {
      safeDelete(outputPath);
      return res.status(500).json({ detail: result.error });
    }

    // Stream output video as download
    res.download(outputPath, `annotated_${Math.floor(Date.now() / 1000)}.mp4`, (err) => {
      // Cleanup the generated video file after download finishes/cancels
      safeDelete(outputPath);
      if (err && !res.headersSent) {
        console.error('Error streaming download:', err);
      }
    });

  } catch (err) {
    safeDelete(inputPath);
    safeDelete(outputPath);
    console.error('Video export execution failed:', err);
    return res.status(500).json({ detail: err.message });
  }
});

// 3. GET /api/health
app.get('/api/health', async (req, res) => {
  try {
    const result = await runCommand({ action: 'health' });
    if (result.status === 'error') {
      return res.status(503).json({ error: result.error });
    }

    // Add additional backend status info
    const healthInfo = {
      ...result.data,
      worker_alive: worker !== null && !worker.killed,
      is_ready: isReady
    };

    return res.json(healthInfo);
  } catch (err) {
    return res.status(503).json({
      status: 'error',
      message: 'Worker unreachable',
      error: err.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`Express server running on http://localhost:${PORT}`);
});
