import React, { useState, useEffect, useRef } from 'react';
export default function UploadZone({ selectedFile, onFileChange, onMetadataChange }) {
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');
  const [videoMeta, setVideoMeta] = useState(null);
  const [thumbnailUrl, setThumbnailUrl] = useState('');
  const fileInputRef = useRef(null);
  const activeObjUrl = useRef('');

  useEffect(() => {
    return () => {
      if (activeObjUrl.current) {
        URL.revokeObjectURL(activeObjUrl.current);
      }
    };
  }, []);

  const handleFile = (file) => {
    if (!file) return;
    if (activeObjUrl.current) {
      URL.revokeObjectURL(activeObjUrl.current);
      activeObjUrl.current = '';
    }

    const isVideo = file.type.startsWith('video/');
    onFileChange(file, isVideo);

    if (isVideo) {
      const objUrl = URL.createObjectURL(file);
      activeObjUrl.current = objUrl;

      // Extract metadata & thumbnail via canvas
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.src = objUrl;

      video.addEventListener('loadeddata', () => {
        video.currentTime = 0;
      });

      video.addEventListener('seeked', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        try {
          const thumbUrl = canvas.toDataURL('image/jpeg', 0.8);
          setThumbnailUrl(thumbUrl);
        } catch (e) {
          console.warn('Could not extract thumbnail, CORS or other issue:', e);
        }

        const duration = video.duration ? `${video.duration.toFixed(1)}s` : '—';
        const size_mb = (file.size / 1024 / 1024).toFixed(1);
        const metadata = {
          width: video.videoWidth,
          height: video.videoHeight,
          duration,
          size_mb,
          type: file.type || 'video'
        };

        setVideoMeta(metadata);
        onMetadataChange(metadata);
      });

      video.load();
      setPreviewUrl('');
    } else {
      // Image reader
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreviewUrl(e.target.result);
        setThumbnailUrl('');
        setVideoMeta(null);
        onMetadataChange(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const onDragLeave = () => {
    setDragOver(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const onFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  };

  const triggerInput = () => {
    fileInputRef.current.click();
  };

  const hasMedia = previewUrl || thumbnailUrl;

  return (
    <div className="upload-container">
      <div 
        id="dropzone"
        className={`${dragOver ? 'drag-over' : ''} ${hasMedia ? 'has-media' : ''}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={triggerInput}
      >
        {!hasMedia ? (
          <>
            <div className="drop-icon">📷</div>
            <p className="drop-text">Drop image or video here or <span>browse</span></p>
            <p className="drop-text-sub">PNG, JPG, WEBP, MP4, MOV, WEBM, AVI — up to 500 MB</p>
          </>
        ) : (
          <>
            {previewUrl && <img id="preview-img" src={previewUrl} alt="preview" />}
            {thumbnailUrl && <img id="preview-canvas" src={thumbnailUrl} alt="video thumbnail" />}
            <div className="change-overlay">↑ click to change file</div>
          </>
        )}
      </div>

      <input 
        type="file" 
        id="file-input" 
        ref={fileInputRef}
        accept="image/*,video/*"
        onChange={onFileSelect}
        style={{ display: 'none' }}
      />

      {selectedFile && videoMeta && (
        <div id="video-meta">
          <span>{videoMeta.width}×{videoMeta.height}</span> &nbsp;|&nbsp;
          <span>{videoMeta.duration}</span> &nbsp;|&nbsp;
          <span>{videoMeta.size_mb} MB</span> &nbsp;|&nbsp;
          <span>{videoMeta.type}</span>
        </div>
      )}
    </div>
  );
}
