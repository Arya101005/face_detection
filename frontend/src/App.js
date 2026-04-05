import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_URL = 'http://localhost:5000';

function App() {
  const [mode, setMode] = useState('upload');
  const [videoFile, setVideoFile] = useState(null);
  const [faceCount, setFaceCount] = useState(0);
  const [persons, setPersons] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const videoRef = useRef(null);

  useEffect(() => {
    checkBackend();
    fetchPersons();
    const interval = setInterval(() => {
      fetchCount();
      fetchPersons();
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const checkBackend = async () => {
    try {
      const res = await fetch(`${API_URL}/count`);
      if (res.ok) {
        setBackendStatus('connected');
      } else {
        setBackendStatus('error');
      }
    } catch (err) {
      setBackendStatus('offline');
    }
  };

  const fetchCount = async () => {
    try {
      const res = await fetch(`${API_URL}/count`);
      const data = await res.json();
      setFaceCount(data.count);
      setError(null);
    } catch (err) {
      setError('Failed to connect to backend');
    }
  };

  const fetchPersons = async () => {
    try {
      const res = await fetch(`${API_URL}/persons`);
      const data = await res.json();
      setPersons(data.persons || []);
    } catch (err) {
      console.error('Error fetching persons:', err);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile(file);
      setPersons([]);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!videoFile) return;

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('video', videoFile);

    try {
      await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      fetchPersons();
    } catch (err) {
      setError('Error uploading video');
    }
    setUploading(false);
  };

  const startCamera = async () => {
    try {
      await fetch(`${API_URL}/reset`, { method: 'POST' });
      setCameraActive(true);
      setPersons([]);
    } catch (err) {
      setError('Failed to start camera');
    }
  };

  const stopCamera = () => {
    setCameraActive(false);
  };

  const handleReset = async () => {
    try {
      await fetch(`${API_URL}/reset`, { method: 'POST' });
      setFaceCount(0);
      setPersons([]);
      setVideoFile(null);
      setError(null);
    } catch (err) {
      setError('Error resetting');
    }
  };

  const getVideoSrc = () => {
    if (!videoFile) return '';
    return `${API_URL}/video_feed?filename=${encodeURIComponent(videoFile.name)}&mode=upload`;
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Face Detection & Tracking</h1>
      
      {backendStatus === 'offline' && (
        <div className="stats-container" style={{background: '#ff4757'}}>
          <p>Backend is offline. Please start the backend server (and MongoDB).</p>
        </div>
      )}

      {error && (
        <div className="stats-container" style={{background: '#ff4757'}}>
          <p>{error}</p>
        </div>
      )}

      <div className="mode-buttons">
        <button 
          onClick={() => setMode('upload')} 
          className={`mode-btn ${mode === 'upload' ? 'active' : ''}`}
        >
          Video Upload
        </button>
        <button 
          onClick={() => setMode('camera')} 
          className={`mode-btn ${mode === 'camera' ? 'active' : ''}`}
        >
          Live Camera
        </button>
      </div>

      {mode === 'upload' && (
        <div className="upload-section">
          <form onSubmit={handleUpload} className="upload-form">
            <div className="file-input-wrapper">
              <input
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="file-input"
                id="video-input"
              />
              <label htmlFor="video-input" className="file-label">
                Choose Video File
              </label>
              {videoFile && <p className="file-name">{videoFile.name}</p>}
            </div>
            <button 
              type="submit" 
              disabled={!videoFile || uploading} 
              className="action-btn"
            >
              {uploading ? <><span className="loading-spinner"></span>Starting...</> : 'Start Detection'}
            </button>
          </form>

          {videoFile && (
            <div className="video-container">
              <img
                ref={videoRef}
                src={getVideoSrc()}
                alt="Video Feed"
                className="video-frame"
                onError={(e) => {
                  console.log('Video feed error:', e);
                }}
              />
            </div>
          )}
        </div>
      )}

      {mode === 'camera' && (
        <div className="upload-section">
          {cameraActive ? (
            <div className="camera-section">
              <img
                src={`${API_URL}/camera`}
                alt="Camera Feed"
                className="video-frame"
              />
              <button onClick={stopCamera} className="stop-btn">
                Stop Camera
              </button>
            </div>
          ) : (
            <button onClick={startCamera} className="action-btn">
              Start Camera
            </button>
          )}
        </div>
      )}

      <div className="stats-container">
        <h2 className="stats-title">Unique Persons Detected</h2>
        <p className="stats-count">{faceCount}</p>
      </div>

      <div style={{display: 'flex', justifyContent: 'center'}}>
        <button onClick={handleReset} className="reset-btn">
          Reset
        </button>
      </div>

      <div className="faces-gallery">
        <h3 className="faces-title">Detected Persons</h3>
        {persons.length === 0 ? (
          <p style={{textAlign: 'center', color: '#888', padding: '20px'}}>No faces detected yet</p>
        ) : (
          <div className="faces-grid">
            {persons.map((person) => (
              <div key={person.name} className="face-card">
                {person.image && (
                  <img
                    src={`${API_URL}/captured_faces/${person.image}`}
                    alt={person.name}
                    className="face-img"
                  />
                )}
                <p className="face-label">{person.name}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;