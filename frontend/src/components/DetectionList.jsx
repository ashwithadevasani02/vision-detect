import React from 'react';

// Earth-toned green, sage, mint, and soft warm colors
const COLORS = ["#2e603a", "#5d8b67", "#388e3c", "#8bc34a", "#009688", "#00acc1", "#81c784", "#afb42b"];

export default function DetectionList({ detections = [] }) {
  const total = detections.length;
  
  const confs = detections.map(d => d.confidence);
  const avgConf = confs.length 
    ? (confs.reduce((a, b) => a + b, 0) / confs.length * 100).toFixed(1) + '%' 
    : '—';
    
  const uniqueClasses = new Set(detections.map(d => d.class)).size;

  return (
    <div className="card" id="detections-card" style={{ display: 'block', gridColumn: '1 / -1' }}>
      <p className="card-label">// 03 — Detection Results</p>
      
      <div className="stats-row">
        <div className="stat">
          <div className="stat-val">{total}</div>
          <div className="stat-lbl">OBJECTS FOUND</div>
        </div>
        <div className="stat">
          <div className="stat-val">{uniqueClasses || '—'}</div>
          <div className="stat-lbl">UNIQUE CLASSES</div>
        </div>
        <div className="stat">
          <div className="stat-val">{avgConf}</div>
          <div className="stat-lbl">AVG CONFIDENCE</div>
        </div>
      </div>

      <div className="det-list">
        {total === 0 ? (
          <p style={{ color: 'var(--muted)', fontFamily: 'var(--mono)', fontSize: '13px' }}>
            No objects detected. Try adjusting the input file.
          </p>
        ) : (
          detections.map((d, i) => {
            const color = COLORS[i % COLORS.length];
            const [x1, y1, x2, y2] = d.bbox || [0, 0, 0, 0];
            return (
              <div 
                key={i} 
                className="det-item" 
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <div className="det-color" style={{ backgroundColor: color }}></div>
                <span className="det-name">{d.class}</span>
                <span className="det-conf">{(d.confidence * 100).toFixed(1)}%</span>
                {d.frame !== null && d.frame !== undefined && (
                  <span className="det-frame">frame {d.frame}</span>
                )}
                <span className="det-bbox">
                  [{x1.toFixed(0)}, {y1.toFixed(0)}, {x2.toFixed(0)}, {y2.toFixed(0)}]
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
