(function () {
  'use strict';

  var video    = document.getElementById('video');
  var startBtn = document.getElementById('startBtn');
  var videoBtn = document.getElementById('videoBtn');
  var statsBtn = document.getElementById('statsBtn');
  var statsPanel = document.getElementById('statsPanel');
  var statsBackdrop = document.getElementById('statsBackdrop');
  var statsClose = document.getElementById('statsClose');
  var statusEl = document.getElementById('status');
  var liveIndicator = document.getElementById('liveIndicator');

  // Panel images
  var panelImgs = {
    edge: document.getElementById('img-edge'),
    sift: document.getElementById('img-sift'),
    cnn:  document.getElementById('img-cnn'),
    yolo: document.getElementById('img-yolo'),
  };

  // Panel footer metrics
  var footer = {
    edgeDensity:  document.getElementById('m-edge-density'),
    edgeTime:     document.getElementById('m-edge-time'),
    siftKp:       document.getElementById('m-sift-kp'),
    siftTracked:  document.getElementById('m-sift-tracked'),
    siftTime:     document.getElementById('m-sift-time'),
    cnnLabel:     document.getElementById('m-cnn-label'),
    cnnConf:      document.getElementById('m-cnn-conf'),
    cnnTime:      document.getElementById('m-cnn-time'),
    yoloDet:      document.getElementById('m-yolo-det'),
    yoloConf:     document.getElementById('m-yolo-conf'),
    yoloTime:     document.getElementById('m-yolo-time'),
  };

  // Stats panel metrics
  var stats = {
    edgeDensity:  document.getElementById('s-edge-density'),
    edgeTime:     document.getElementById('s-edge-time'),
    siftKp:       document.getElementById('s-sift-kp'),
    siftTracked:  document.getElementById('s-sift-tracked'),
    siftTime:     document.getElementById('s-sift-time'),
    cnnLabel:     document.getElementById('s-cnn-label'),
    cnnConf:      document.getElementById('s-cnn-conf'),
    cnnTime:      document.getElementById('s-cnn-time'),
    yoloDet:      document.getElementById('s-yolo-det'),
    yoloConf:     document.getElementById('s-yolo-conf'),
    yoloTime:     document.getElementById('s-yolo-time'),
  };

  // Timing bars (inside stats panel)
  var bars = {
    edge: { bar: document.getElementById('bar-edge'), val: document.getElementById('bar-edge-val') },
    sift: { bar: document.getElementById('bar-sift'), val: document.getElementById('bar-sift-val') },
    cnn:  { bar: document.getElementById('bar-cnn'),  val: document.getElementById('bar-cnn-val')  },
    yolo: { bar: document.getElementById('bar-yolo'), val: document.getElementById('bar-yolo-val') },
  };

  // Perturbation controls
  var perturbTypes = ['noise', 'blur', 'rotation', 'brightness'];
  var perturbControls = {};
  perturbTypes.forEach(function (t) {
    perturbControls[t] = {
      checkbox:   document.getElementById('chk-' + t),
      slider:     document.getElementById('slider-' + t),
      valDisplay: document.getElementById('val-' + t),
    };
  });

  var resetBtn = document.getElementById('resetBtn');

  var VIDEO_SRC = '/public/sift.mp4';

  // State
  var stream     = null;
  var videoMode  = false;   // true when playing the pre-recorded video
  var canvas     = null;
  var ctx        = null;
  var running    = false;
  var pending    = false;
  var CAPTURE_W  = 320;
  var CAPTURE_H  = 240;
  var TARGET_FPS = 5;
  var lastSendTime = 0;
  var frameCount   = 0;
  var errorCount   = 0;

  // ── Stats Panel Toggle ──────────────────────────────────────

  function openStats() {
    statsPanel.classList.add('open');
    statsBackdrop.classList.add('open');
    statsBtn.textContent = 'Hide Stats';
  }

  function closeStats() {
    statsPanel.classList.remove('open');
    statsBackdrop.classList.remove('open');
    statsBtn.textContent = 'Stats';
  }

  statsBtn.addEventListener('click', function () {
    statsPanel.classList.contains('open') ? closeStats() : openStats();
  });
  statsClose.addEventListener('click', closeStats);
  statsBackdrop.addEventListener('click', closeStats);

  // ── Perturbation helpers ────────────────────────────────────

  function getPerturbationParams() {
    var params = {};
    perturbTypes.forEach(function (t) {
      var c = perturbControls[t];
      params[t] = { active: c.checkbox.checked, value: Number(c.slider.value) };
    });
    return params;
  }

  perturbTypes.forEach(function (t) {
    var c = perturbControls[t];
    c.slider.addEventListener('input', function () {
      c.valDisplay.textContent = c.slider.value;
      if (!c.checkbox.checked) c.checkbox.checked = true;
    });
  });

  resetBtn.addEventListener('click', function () {
    perturbTypes.forEach(function (t) {
      var c = perturbControls[t];
      c.checkbox.checked = false;
      c.slider.value = c.slider.min === '-180' ? '0' : c.slider.min;
      c.valDisplay.textContent = c.slider.value;
    });
  });

  // ── Helpers ─────────────────────────────────────────────────

  function setText(el, val) { if (el) el.textContent = val; }

  function setStatus(text, isError) {
    statusEl.textContent = text;
    statusEl.className = 'status' + (isError ? ' error' : '');
  }

  // ── Update UI ───────────────────────────────────────────────

  function updatePanels(panels) {
    Object.keys(panels).forEach(function (key) {
      var img = panelImgs[key];
      if (!img) return;
      img.src = 'data:image/jpeg;base64,' + panels[key];
      // Hide the "No signal" placeholder once we have real data
      var empty = img.parentElement && img.parentElement.querySelector('.panel-empty');
      if (empty) empty.style.display = 'none';
    });
  }

  function updateMetrics(metrics) {
    var e = metrics.edge || {};
    var s = metrics.sift || {};
    var c = metrics.cnn  || {};
    var y = metrics.yolo || {};

    var edgeDensity = (e.edge_density    || 0).toFixed(1) + '%';
    var edgeTime    = (e.time_ms         || 0).toFixed(1) + 'ms';
    var siftKp      = Math.round(s.keypoints_filtered || 0);
    var siftTracked = Math.round(s.matches_good       || 0);
    var siftTime    = (s.time_ms         || 0).toFixed(1) + 'ms';
    var rawLabel    = c.top1_label || '—';
    var shortLabel  = rawLabel.length > 14 ? rawLabel.substring(0, 14) + '\u2026' : rawLabel;
    var cnnConf     = (c.top1_confidence || 0).toFixed(1) + '%';
    var cnnTime     = (c.time_ms         || 0).toFixed(1) + 'ms';
    var yoloDet     = y.detections || 0;
    var yoloConf    = (y.person_confidence || 0).toFixed(1) + '%';
    var yoloTime    = (y.time_ms           || 0).toFixed(1) + 'ms';

    // Panel footers
    setText(footer.edgeDensity, edgeDensity);
    setText(footer.edgeTime,    edgeTime);
    setText(footer.siftKp,      siftKp);
    setText(footer.siftTracked, siftTracked);
    setText(footer.siftTime,    siftTime);
    setText(footer.cnnLabel,    shortLabel);
    setText(footer.cnnConf,     cnnConf);
    setText(footer.cnnTime,     cnnTime);
    setText(footer.yoloDet,     yoloDet);
    setText(footer.yoloConf,    yoloConf);
    setText(footer.yoloTime,    yoloTime);

    // Stats overlay (full label here)
    setText(stats.edgeDensity, edgeDensity);
    setText(stats.edgeTime,    edgeTime);
    setText(stats.siftKp,      siftKp);
    setText(stats.siftTracked, siftTracked);
    setText(stats.siftTime,    siftTime);
    setText(stats.cnnLabel,    rawLabel);
    setText(stats.cnnConf,     cnnConf);
    setText(stats.cnnTime,     cnnTime);
    setText(stats.yoloDet,     yoloDet);
    setText(stats.yoloConf,    yoloConf);
    setText(stats.yoloTime,    yoloTime);

    // Timing bars
    var times = {
      edge: e.time_ms || 0,
      sift: s.time_ms || 0,
      cnn:  c.time_ms || 0,
      yolo: y.time_ms || 0,
    };
    var maxTime = Math.max(times.edge, times.sift, times.cnn, times.yolo, 1);
    Object.keys(bars).forEach(function (key) {
      bars[key].bar.style.width = ((times[key] / maxTime) * 100) + '%';
      bars[key].val.textContent = times[key].toFixed(1) + 'ms';
    });
  }

  // ── Capture loop ────────────────────────────────────────────

  function loop() {
    if (!running) return;
    requestAnimationFrame(loop);

    if (pending) return;
    if (!video.videoWidth) return;

    var now = performance.now();
    if (now - lastSendTime < 1000 / TARGET_FPS) return;
    lastSendTime = now;

    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.width  = CAPTURE_W;
      canvas.height = CAPTURE_H;
      ctx = canvas.getContext('2d');
    }

    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);

    pending = true;
    canvas.toBlob(function (blob) {
      if (!blob) { pending = false; return; }
      sendFrame(blob);
    }, 'image/jpeg', 0.85);
  }

  function sendFrame(blob) {
    var fd = new FormData();
    fd.append('file', blob, 'frame.jpg');
    fd.append('perturbations', JSON.stringify(getPerturbationParams()));

    fetch('/process-frame', { method: 'POST', body: fd })
      .then(function (res) {
        if (!res.ok) {
          return res.text().then(function (t) {
            throw new Error('Server ' + res.status + ': ' + t.substring(0, 80));
          });
        }
        return res.json();
      })
      .then(function (data) {
        updatePanels(data.panels);
        updateMetrics(data.metrics);
        frameCount++;
        errorCount = 0;
        if (frameCount === 1) liveIndicator.classList.add('visible');
        setStatus((videoMode ? 'Video' : 'Live') + ' \u2014 frame ' + frameCount);
      })
      .catch(function (err) {
        errorCount++;
        setStatus(err.message, true);
        console.error('[cv] process-frame error:', err);
        if (errorCount > 5) lastSendTime = performance.now() + 2000;
      })
      .finally(function () { pending = false; });
  }

  // ── Stop helper (shared) ─────────────────────────────────────

  function stopAll() {
    running = false;
    videoMode = false;
    liveIndicator.classList.remove('visible');
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    video.pause();
    video.removeAttribute('src');
    video.srcObject = null;
    startBtn.textContent = '\uD83D\uDCF7 Camera';
    videoBtn.textContent = '\u25B6 Video';
    videoBtn.classList.remove('active');
    startBtn.disabled = false;
    videoBtn.disabled = false;
    setStatus('Stopped');
  }

  function startLoop() {
    frameCount   = 0;
    errorCount   = 0;
    lastSendTime = 0;
    pending      = false;
    running      = true;
    setStatus('Connecting\u2026');
    loop();
  }

  // ── Camera ───────────────────────────────────────────────────

  startBtn.addEventListener('click', function () {
    if (stream || videoMode) { stopAll(); return; }

    setStatus('Requesting camera access\u2026');
    startBtn.disabled = true;
    videoBtn.disabled = true;

    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then(function (s) {
        stream = s;
        video.srcObject = s;
        video.onloadedmetadata = function () {
          video.play().then(function () {
            startBtn.textContent = '\u23F9 Stop';
            startBtn.disabled = false;
            videoBtn.disabled = false;
            startLoop();
          }).catch(function (err) {
            setStatus('Could not play video: ' + err.message, true);
            startBtn.disabled = false;
            videoBtn.disabled = false;
          });
        };
      })
      .catch(function (err) {
        setStatus('Camera access denied.', true);
        console.error('[cv] getUserMedia error:', err);
        startBtn.disabled = false;
        videoBtn.disabled = false;
      });
  });

  // ── Video file ───────────────────────────────────────────────

  videoBtn.addEventListener('click', function () {
    if (videoMode) { stopAll(); return; }
    if (stream)    { stopAll(); }

    videoBtn.disabled = true;
    startBtn.disabled = true;
    setStatus('Loading video\u2026');

    video.srcObject = null;
    video.src = VIDEO_SRC;
    video.loop = true;

    video.addEventListener('canplay', function onCanPlay() {
      video.removeEventListener('canplay', onCanPlay);
      video.play().then(function () {
        videoMode = true;
        videoBtn.textContent = '\u23F9 Stop Video';
        videoBtn.classList.add('active');
        videoBtn.disabled = false;
        startBtn.disabled = false;
        startLoop();
      }).catch(function (err) {
        setStatus('Could not play video: ' + err.message, true);
        videoBtn.disabled = false;
        startBtn.disabled = false;
      });
    });

    video.addEventListener('error', function () {
      setStatus('Video not found — place sift.mp4 in web/public/', true);
      videoBtn.disabled = false;
      startBtn.disabled = false;
    }, { once: true });

    video.load();
  });

  // Export for external use if needed
  window.cvDemo = {
    hasStream:   function () { return !!(stream || videoMode); },
    isVideoMode: function () { return videoMode; },
    setStatus:   setStatus,
  };

})();
