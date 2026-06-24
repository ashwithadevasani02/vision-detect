import React, { useEffect, useRef } from 'react';

// Earth-toned green, sage, mint, and soft warm colors
const COLORS = ["#2e603a", "#5d8b67", "#388e3c", "#8bc34a", "#009688", "#00acc1", "#81c784", "#afb42b"];

export default function VideoPlayer({ src, detections = [], fps = 30 }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const context = canvas.getContext('2d');

    const resizeCanvas = () => {
      canvas.width = video.videoWidth || 1;
      canvas.height = video.videoHeight || 1;
    };

    const pickFrameDetections = (frameIndex) => {
      if (!detections.length) return [];
      
      let bestFrame = detections[0].frame ?? 0;
      let bestDiff = Math.abs(bestFrame - frameIndex);
      
      for (const detection of detections) {
        const frame = detection.frame ?? 0;
        const diff = Math.abs(frame - frameIndex);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestFrame = frame;
        }
      }
      return detections.filter(detection => (detection.frame ?? 0) === bestFrame);
    };

    const draw = () => {
      if (!video || !canvas) return;

      if (video.videoWidth && canvas.width !== video.videoWidth) {
        resizeCanvas();
      }

      context.clearRect(0, 0, canvas.width, canvas.height);
      
      if (video.paused || video.ended) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const currentFrame = Math.floor((video.currentTime || 0) * fps);
      const frameDetections = pickFrameDetections(currentFrame);

      context.lineWidth = Math.max(2, Math.round(canvas.width / 320));
      context.font = `${Math.max(14, Math.round(canvas.width / 40))}px Plus Jakarta Sans, sans-serif`;
      
      frameDetections.forEach((detection, index) => {
        const color = COLORS[index % COLORS.length];
        const bbox = detection.bbox || [0, 0, canvas.width, canvas.height];
        const [x1, y1, x2, y2] = bbox;
        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;
        const label = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
        
        const labelWidth = context.measureText(label).width + 10;
        
        context.strokeStyle = color;  
        context.fillStyle = color;
        
        // Draw rectangle box
        context.strokeRect(x1, y1, boxWidth, boxHeight);
        
        // Draw text label background
        context.fillRect(x1, Math.max(0, y1 - 22), labelWidth, 22);
        
        // Draw text label
        context.fillStyle = "#ffffff";
        context.fillText(label, x1 + 5, Math.max(16, y1 - 6));
      });

      rafRef.current = requestAnimationFrame(draw);
    };

    video.addEventListener("loadedmetadata", resizeCanvas);
    
    const startLoop = () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      draw();
    };

    video.addEventListener("play", startLoop);
    video.addEventListener("seeked", draw);
    video.addEventListener("pause", draw);

    resizeCanvas();
    draw();

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      video.removeEventListener("loadedmetadata", resizeCanvas);
      video.removeEventListener("play", startLoop);
      video.removeEventListener("seeked", draw);
      video.removeEventListener("pause", draw);
    };
  }, [src, detections, fps]);

  return (
    <div id="result-media-wrap">
      <video 
        ref={videoRef}
        id="result-video" 
        src={src} 
        controls 
        autoPlay 
        muted 
        loop 
        playsInline 
        style={{ display: 'block' }}
      />
      <canvas 
        ref={canvasRef}
        id="result-video-overlay" 
        style={{ display: 'block' }}
      />
    </div>
  );
}
