import React, { useState, useEffect, useRef } from 'react';
import UploadZone from './components/UploadZone';
import DetectionList from './components/DetectionList';
import VideoPlayer from './components/VideoPlayer';

const API_HOST = 'http://localhost:5005'; // Updated to matching backend port 5005

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isVideo, setIsVideo] = useState(false);
  const [videoMeta, setVideoMeta] = useState(null);
  
  // Settings are hardcoded and simplified (hidden from UI)
  const [settings] = useState({
    frame_stride: 30,
    max_frames: 10,
    conf_threshold: 0.25,
    iou_threshold: 0.5,
  });

  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [progressPct, setProgressPct] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const [error, setError] = useState('');
  
  // Results
  const [hasResults, setHasResults] = useState(false);
  const [detections, setDetections] = useState([]);
  const [rawDetections, setRawDetections] = useState([]);
  const [annotatedImage, setAnnotatedImage] = useState('');
  const [apiVideoFps, setApiVideoFps] = useState(30);

  const progressInterval = useRef(null);
  const fileUrl = useRef('');

  useEffect(() => {
    return () => {
      if (fileUrl.current) {
        URL.revokeObjectURL(fileUrl.current);
      }
      clearInterval(progressInterval.current);
    };
  }, []);

  const handleFileChange = (file, isVideoFile) => {
    setSelectedFile(file);
    setIsVideo(isVideoFile);
    
    // Revoke old file URL
    if (fileUrl.current) {
      URL.revokeObjectURL(fileUrl.current);
      fileUrl.current = '';
    }

    if (file) {
      fileUrl.current = URL.createObjectURL(file);
    }

    // Reset results and errors on new file select
    setHasResults(false);
    setDetections([]);
    setRawDetections([]);
    setAnnotatedImage('');
    setError('');
  };

  const handleReset = () => {
    if (fileUrl.current) {
      URL.revokeObjectURL(fileUrl.current);
      fileUrl.current = '';
    }
    setSelectedFile(null);
    setIsVideo(false);
    setVideoMeta(null);
    setHasResults(false);
    setDetections([]);
    setRawDetections([]);
    setAnnotatedImage('');
    setError('');
    setProgressPct(0);
    setProgressLabel('');
    setIsLoading(false);
    setIsExporting(false);
  };

  const startFakeProgress = () => {
    setProgressPct(0);
    setProgressLabel(isVideo ? 'Uploading video…' : 'Processing…');
    
    let pct = 0;
    progressInterval.current = setInterval(() => {
      const step = pct < 60 ? 2 : pct < 85 ? 0.5 : 0.1;
      pct = Math.min(90, pct + step);
      const label = pct < 40
        ? (isVideo ? 'Uploading video…' : 'Uploading…')
        : pct < 70
          ? 'Running inference on frames…'
          : 'Merging detections…';
      setProgressPct(pct);
      setProgressLabel(label);
    }, 300);
  };

  const stopFakeProgress = () => {
    clearInterval(progressInterval.current);
    setProgressPct(100);
    setProgressLabel('Done!');
    setTimeout(() => {
      setProgressPct(0);
      setProgressLabel('');
    }, 800);
  };

  const runDetection = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError('');
    setHasResults(false);
    startFakeProgress();

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('frame_stride', settings.frame_stride);
    formData.append('max_frames', settings.max_frames);
    formData.append('conf_threshold', settings.conf_threshold);
    formData.append('iou_threshold', settings.iou_threshold);

    try {
      const response = await fetch(`${API_HOST}/api/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: `Server error ${response.status}` }));
        throw new Error(errData.detail || `Server error ${response.status}`);
      }

      const data = await response.json();
      stopFakeProgress();

      setDetections(data.detections || []);
      setRawDetections(data.raw_detections || []);
      setAnnotatedImage(data.annotated_image || '');
      setApiVideoFps(data.video_fps || 30);
      setHasResults(true);

      // Smooth scroll to results
      setTimeout(() => {
        const card = document.getElementById('detections-card');
        if (card) {
          card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }, 100);

    } catch (err) {
      clearInterval(progressInterval.current);
      setProgressPct(0);
      setProgressLabel('');
      setError(`⚠ ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadAnnotatedVideo = async () => {
    if (!selectedFile || !hasResults || !isVideo) return;

    setIsExporting(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('detections_json', JSON.stringify(rawDetections));
    formData.append('fps', apiVideoFps);

    try {
      const response = await fetch(`${API_HOST}/api/download-annotated-video`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: `Video export error ${response.status}` }));
        throw new Error(errData.detail || `Video export error ${response.status}`);
      }

      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = `annotated_${Math.floor(Date.now() / 1000)}.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(downloadUrl);

    } catch (err) {
      setError(`⚠ Video export failed: ${err.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="wrapper">
      <header>
        <h1>VisionDetect</h1>
        <p className="subtitle">Select an image or high-resolution video. The AI model will identify and locate objects instantly.</p>
      </header>

      <div className="grid">
        {/* Input Card */}
        <div className="card">
          <p className="card-label">Input Media</p>

          <UploadZone 
            selectedFile={selectedFile}
            onFileChange={handleFileChange}
            onMetadataChange={setVideoMeta}
          />

          {progressPct > 0 && (
            <div id="progress-wrap" style={{ display: 'block' }}>
              <p className="progress-label">{progressLabel}</p>
              <div className="progress-track">
                <div className="progress-bar" style={{ width: `${progressPct}%` }}></div>
              </div>
            </div>
          )}

          <div className="btn-row">
            <button 
              className="btn" 
              onClick={runDetection}
              disabled={!selectedFile || isLoading}
            >
              <span className="shimmer"></span>
              {isLoading ? (
                <>
                  <span className="spinner"></span>
                  Detecting…
                </>
              ) : (
                'Detect Objects'
              )}
            </button>

            {selectedFile && (
              <button className="btn-ghost visible" onClick={handleReset}>
                ↺ New
              </button>
            )}
          </div>

          {error && <div className="error-box" style={{ display: 'block' }}>{error}</div>}
        </div>

        {/* Bounding Box / Result Card */}
        <div className="card" id="result-card" style={{ display: hasResults ? 'block' : 'none' }}>
          <p className="card-label">Annotated Output</p>
          
          {!isVideo && annotatedImage && (
            <img id="result-img" src={annotatedImage} alt="annotated result" />
          )}

          {isVideo && hasResults && fileUrl.current && (
            <VideoPlayer 
              src={fileUrl.current}
              detections={rawDetections}
              fps={apiVideoFps}
            />
          )}

          {isVideo && hasResults && (
            <button 
              className="btn" 
              style={{ marginTop: '20px' }} 
              onClick={downloadAnnotatedVideo}
              disabled={isExporting}
            >
              <span className="shimmer"></span>
              {isExporting ? (
                <>
                  <span className="spinner"></span>
                  Exporting Video…
                </>
              ) : (
                '↓ Download Annotated Video'
              )}
            </button>
          )}
        </div>

        {/* Detections List */}
        {hasResults && (
          <DetectionList detections={detections} />
        )}
      </div>
    </div>
  );
}
