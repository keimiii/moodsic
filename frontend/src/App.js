import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import './index.css';

function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [emotionData, setEmotionData] = useState(null);
  const [clusters, setClusters] = useState([]);
  const [currentSong, setCurrentSong] = useState(null);
  const [audioElement, setAudioElement] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  // Load clusters and videos on component mount
  useEffect(() => {
    loadClusters();
    loadVideos();
  }, []);

  // Load clusters from backend
  const loadClusters = async () => {
    try {
      const response = await axios.get('/api/clusters');
      setClusters(response.data);
    } catch (error) {
      console.error('Error loading clusters:', error);
    }
  };

  // Load videos from backend
  const loadVideos = async () => {
    try {
      const response = await axios.get('/api/videos');
      setVideos(response.data);
    } catch (error) {
      console.error('Error loading videos:', error);
    }
  };

  // Handle video selection
  const handleVideoSelect = (video) => {
    setSelectedVideo(video);
    setVideoUrl(`/api/video/${video.id}`);
    setEmotionData(null); // Reset emotion data when selecting new video
    if (audioElement) {
      audioElement.pause();
    }
    setIsPlaying(false);
    setCurrentSong(null);
    setCurrentTime(0);
    setDuration(0);
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  };

  // Process video for emotion analysis
  const processVideo = async () => {
    if (!selectedVideo) return;

    setIsProcessing(true);
    try {
      const response = await axios.post('/api/process-video', {
        video_id: selectedVideo.id
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      setEmotionData(response.data);
      
      // Start playing the recommended song
      if (response.data.song) {
        playSong(response.data.song);
      }
    } catch (error) {
      console.error('Error processing video:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Play recommended song
  const playSong = (song) => {
    if (audioElement) {
      audioElement.pause();
    }

    const audio = new Audio(`/api/song/${song.song_id}`);
    audio.addEventListener('loadedmetadata', () => {
      setDuration(audio.duration);
    });
    
    audio.addEventListener('timeupdate', () => {
      setCurrentTime(audio.currentTime);
    });

    audio.addEventListener('ended', () => {
      setIsPlaying(false);
      if (videoRef.current && !videoRef.current.paused) {
        videoRef.current.pause();
      }
    });

    const startMediaPlayback = () => {
      if (videoRef.current) {
        try {
          videoRef.current.currentTime = 0;
          const videoPromise = videoRef.current.play();
          if (videoPromise && typeof videoPromise.catch === 'function') {
            videoPromise.catch((error) => {
              console.warn('Video playback blocked:', error);
            });
          }
        } catch (error) {
          console.warn('Unable to start video playback:', error);
        }
      }

      const audioPromise = audio.play();
      if (audioPromise && typeof audioPromise.then === 'function') {
        audioPromise
          .then(() => setIsPlaying(true))
          .catch((error) => {
            console.warn('Audio playback blocked:', error);
            setIsPlaying(false);
          });
      } else {
        setIsPlaying(true);
      }
    };

    setAudioElement(audio);
    setCurrentSong(song);
    setCurrentTime(0);

    if (audio.readyState >= 1) {
      startMediaPlayback();
    } else {
      const handleCanPlay = () => {
        audio.removeEventListener('canplay', handleCanPlay);
        startMediaPlayback();
      };
      audio.addEventListener('canplay', handleCanPlay);
      audio.load();
    }
  };

  // Toggle play/pause
  const togglePlayPause = () => {
    if (audioElement) {
      if (isPlaying) {
        audioElement.pause();
        if (videoRef.current && !videoRef.current.paused) {
          videoRef.current.pause();
        }
        setIsPlaying(false);
      } else {
        if (videoRef.current) {
          try {
            const videoPromise = videoRef.current.play();
            if (videoPromise && typeof videoPromise.catch === 'function') {
              videoPromise.catch((error) => {
                console.warn('Video playback blocked:', error);
              });
            }
          } catch (error) {
            console.warn('Unable to resume video playback:', error);
          }
        }
        const audioPromise = audioElement.play();
        if (audioPromise && typeof audioPromise.then === 'function') {
          audioPromise
            .then(() => setIsPlaying(true))
            .catch((error) => {
              console.warn('Audio playback blocked:', error);
              setIsPlaying(false);
            });
        } else {
          setIsPlaying(true);
        }
      }
    }
  };

  const handleVideoPlay = () => {
    if (audioElement && audioElement.paused) {
      const audioPromise = audioElement.play();
      if (audioPromise && typeof audioPromise.catch === 'function') {
        audioPromise.catch((error) => {
          console.warn('Audio playback blocked:', error);
        });
      } else {
        setIsPlaying(true);
      }
    }
  };

  const handleVideoPause = () => {
    if (audioElement && !audioElement.paused) {
      audioElement.pause();
      setIsPlaying(false);
    }
  };

  // Format time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatMae = (value) => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value.toFixed(3);
    }
    return '—';
  };

  const bestMaePathway = useMemo(() => {
    if (!emotionData?.mae) {
      return null;
    }

    let bestPathway = null;
    let bestScore = Number.POSITIVE_INFINITY;

    ['scene', 'face', 'fusion'].forEach((pathway) => {
      const metrics = emotionData.mae[pathway];
      if (!metrics) {
        return;
      }

      const scores = [metrics.valence, metrics.arousal].filter(
        (value) => typeof value === 'number' && Number.isFinite(value)
      );

      if (!scores.length) {
        return;
      }

      const averageMae = scores.reduce((sum, value) => sum + Math.abs(value), 0) / scores.length;

      if (averageMae < bestScore) {
        bestScore = averageMae;
        bestPathway = pathway;
      }
    });

    return bestPathway;
  }, [emotionData]);

  // Canvas animation
  useEffect(() => {
    if (!canvasRef.current || !clusters.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const resizeCanvas = () => {
      const container = canvas.parentElement;
      canvas.width = container.clientWidth - 48;
      canvas.height = container.clientHeight - 48;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const animate = () => {
      ctx.fillStyle = 'rgba(15, 15, 35, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      
      // Vertical lines
      for (let i = 0; i <= 10; i++) {
        const x = (canvas.width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      
      // Horizontal lines
      for (let i = 0; i <= 10; i++) {
        const y = (canvas.height / 10) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // Draw clusters, for each cluster, draw its points with different colours
      const cluster_colours = [
        '255, 99, 132',   // Red
        '54, 162, 235',   // Blue
        '255, 206, 86',   // Yellow
        '75, 192, 192',   // Teal
        '153, 102, 255',  // Purple
      ];
      clusters.forEach((cluster, clusterIndex) => {
        const centerX = ((cluster.center.valence + 1) / 2) * canvas.width;
        const centerY = ((1 - cluster.center.arousal) / 2) * canvas.height;
        
        // Draw all points in the cluster
        if (cluster.points && cluster.points.length > 0) {
          cluster.points.forEach(point => {
            const pointX = ((point.valence + 1) / 2) * canvas.width;
            const pointY = ((1 - point.arousal) / 2) * canvas.height;
            
            const isActive = emotionData && emotionData.cluster_id === cluster.id;
            const alpha = isActive ? 0.9 : 0.03;
            const clusterColour = cluster_colours[clusterIndex % cluster_colours.length];
            
            // Draw each point
            ctx.fillStyle = `rgba(${clusterColour}, ${alpha})`;
            ctx.beginPath();
            ctx.arc(pointX, pointY, 2, 0, Math.PI * 2);
            ctx.fill();
          });
        }

        // Draw animated cluster boundary
        ctx.strokeStyle = emotionData && emotionData.cluster_id === cluster.id 
          ? 'rgba(255, 255, 255, 0.7)'
          : 'rgba(255, 255, 255, 0.1)';
        ctx.beginPath();
        for (let i = 0; i < 50; i++) {
          const angle = (Math.PI * 2 * i) / 50;
          const radius = 20 + Math.sin(Date.now() * 0.001 + angle) * 5;
          const x = centerX + Math.cos(angle) * radius;
          const y = centerY + Math.sin(angle) * radius;
          
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.closePath();
        ctx.stroke();

        // Draw cluster center
        if (emotionData && emotionData.cluster_id === cluster.id) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.beginPath();
          ctx.arc(centerX, centerY, 8, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [clusters, emotionData]);

  return (
    <div className="container">
      <div className="video-panel">
        <div className="header">
          <h1>Moodsic</h1>
          <p className="subtitle">Multimodal MoE Detection</p>
        </div>
        
        <div className="controls">
          <div className="video-selection">
            <h3>Select Video:</h3>
            <select 
              value={selectedVideo?.id || ''} 
              onChange={(e) => {
                const video = videos.find(v => v.id === e.target.value);
                if (video) handleVideoSelect(video);
              }}
              className="video-dropdown"
            >
              <option value="">Choose a video...</option>
              {videos.map((video) => (
                <option key={video.id} value={video.id}>
                  {video.name}
                </option>
              ))}
            </select>
          </div>
          <button 
            onClick={processVideo} 
            disabled={!selectedVideo || isProcessing}
            className="analyze-button"
          >
            {isProcessing ? 'Processing...' : 'Analyze'}
          </button>
        </div>
        
        <div style={{ height: '20px' }}></div>
        
        <div className="video-container">
          <div className="video-wrapper">
            {videoUrl ? (
              <video 
                src={videoUrl} 
                controls 
                ref={videoRef}
                onPlay={handleVideoPlay}
                onPause={handleVideoPause}
                style={{ display: 'block' }}
              />
            ) : (
              <div className="video-placeholder">
                <span>Select a video to analyze</span>
              </div>
            )}
          </div>
          <div className="video-info">
            <div className="video-title">
              {emotionData ? 'Analysis Complete' : 'Video Analysis'}
            </div>
            {emotionData && (
              <div className="model-breakdown">
                <article className="signal-card">
                  <header className="signal-title">Fusion Output</header>
                  <div className="signal-line">
                    <span className="signal-label">Valence</span>
                    <span className="signal-value">
                      {emotionData.valence >= 0 ? '+' : ''}{emotionData.valence.toFixed(2)}
                    </span>
                  </div>
                  <div className="signal-line">
                    <span className="signal-label">Arousal</span>
                    <span className="signal-value">
                      {emotionData.arousal >= 0 ? '+' : ''}{emotionData.arousal.toFixed(2)}
                    </span>
                  </div>
                  <div className="signal-status">Variance-weighted</div>
                </article>
                {emotionData.mae && (
                  <div className="mae-metrics">
                    <div className="mae-header">
                      <span className="mae-heading">Pathway MAE</span>
                      <span className="mae-subheading">Valence · Arousal</span>
                    </div>
                    <div className="mae-grid">
                      {['scene', 'face', 'fusion'].map((pathway) => {
                        const metrics = emotionData.mae[pathway] || {};
                        const title = `${pathway.charAt(0).toUpperCase()}${pathway.slice(1)}`;
                        return (
                          <article
                            key={pathway}
                            className={`mae-card${pathway === bestMaePathway ? ' is-best' : ''}`}
                          >
                            <header className="mae-title">
                              <span>{title}</span>
                              {pathway === bestMaePathway && <span className="mae-badge">Lead</span>}
                            </header>
                            <div className="mae-row">
                              <span className="mae-label">Val</span>
                              <span className="mae-value">{formatMae(metrics.valence)}</span>
                            </div>
                            <div className="mae-row">
                              <span className="mae-label">Aro</span>
                              <span className="mae-value">{formatMae(metrics.arousal)}</span>
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
            <div className="video-description">
              {emotionData 
                ? `Analysis complete. Valence: ${emotionData.valence.toFixed(2)}, Arousal: ${emotionData.arousal.toFixed(2)}`
                : 'Select a video to analyze emotions and get music recommendations.'
              }
            </div>
          </div>
        </div>
      </div>
      
      <div className="viz-panel">
        <div className="viz-container">
          <div className="viz-heading">DEAM dataset: Clusters</div>
          <canvas ref={canvasRef}></canvas>
          <div className="axis-labels x-label">Valence (Negative ← → Positive)</div>
          <div className="axis-labels y-label">Arousal (Calm ← → Excited)</div>
          {emotionData && (
            <div className="cluster-overlay">
              <article className="cluster-card">
                <span className="cluster-label">Cluster {emotionData.cluster_id}</span>
                <h3>{clusters[emotionData.cluster_id]?.name || 'Unknown Cluster'}</h3>
                <p className="cluster-mood">
                  {clusters[emotionData.cluster_id]?.mood || 'No description available'}
                </p>
                <p className="cluster-center">
                  Center V~{clusters[emotionData.cluster_id]?.center?.valence?.toFixed(2) || '0.00'} | 
                  A~{clusters[emotionData.cluster_id]?.center?.arousal?.toFixed(2) || '0.00'}
                </p>
                <p className="cluster-current">
                  Fusion V {emotionData.valence >= 0 ? '+' : ''}{emotionData.valence.toFixed(2)} · 
                  A {emotionData.arousal >= 0 ? '+' : ''}{emotionData.arousal.toFixed(2)}
                </p>
                <ul className="cluster-traits">
                  {clusters[emotionData.cluster_id]?.traits?.map((trait, index) => (
                    <li key={index}>{trait}</li>
                  ))}
                </ul>
              </article>
            </div>
          )}
        </div>
        
        {currentSong && (
          <div className="radio-container">
            <div className="radio-main">
              <div className="album-art"></div>
              <div className="track-info">
                <div className="track-title">{currentSong.title}</div>
                <div className="track-artist">{currentSong.artist}</div>
                <div className="station">Station: Scene → DEAM Match</div>
              </div>
              <div className="equalizer" aria-hidden="true">
                <div className="bar"></div>
                <div className="bar"></div>
                <div className="bar"></div>
              </div>
            </div>
            <div className="progress">
              <div 
                className="progress-fill" 
                style={{ 
                  width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` 
                }}
              ></div>
            </div>
            <div className="timecodes">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
            <div className="controls" style={{ marginTop: '1rem' }}>
              <button onClick={togglePlayPause}>
                {isPlaying ? 'Pause' : 'Play'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
