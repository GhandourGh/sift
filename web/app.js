(function () {
  const video = document.getElementById('video');
  const startBtn = document.getElementById('startBtn');
  const panelGrid = document.getElementById('panelGrid');
  const statusEl = document.getElementById('status');

  const TARGET_FPS = 6;
  const CAPTURE_WIDTH = 320;
  const CAPTURE_HEIGHT = 240;

  let stream = null;
  let canvas = null;
  let ctx = null;
  let animationId = null;
  let lastSendTime = 0;
  let pending = false;
  let frameCount = 0;
  let fpsTime = 0;
  let fpsValue = 0;

  function setStatus(text, isError) {
    statusEl.textContent = text;
    statusEl.className = 'status' + (isError ? ' error' : '');
  }

  function revokePrevUrl() {
    if (panelGrid.src && panelGrid.src.startsWith('blob:')) {
      URL.revokeObjectURL(panelGrid.src);
    }
  }

  function captureAndSend() {
    if (!stream || !video.videoWidth || pending) return;

    const now = performance.now();
    const interval = 1000 / TARGET_FPS;
    if (now - lastSendTime < interval) {
      animationId = requestAnimationFrame(captureAndSend);
      return;
    }
    lastSendTime = now;

    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.width = CAPTURE_WIDTH;
      canvas.height = CAPTURE_HEIGHT;
      ctx = canvas.getContext('2d');
    }
    ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    canvas.toBlob(sendFrame, 'image/jpeg', 0.85);
  }

  function sendFrame(blob) {
    if (!blob) {
      pending = false;
      return;
    }
    pending = true;
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');

    fetch('/process-frame', {
      method: 'POST',
      body: formData,
    })
      .then(function (res) {
        if (!res.ok) throw new Error(res.status === 503 ? 'Server starting…' : 'Request failed: ' + res.status);
        return res.blob();
      })
      .then(function (blob) {
        revokePrevUrl();
        panelGrid.src = URL.createObjectURL(blob);
        frameCount++;
        const now = performance.now();
        if (now - fpsTime >= 1000) {
          fpsValue = frameCount;
          frameCount = 0;
          fpsTime = now;
        }
        setStatus('Live');
      })
      .catch(function (err) {
        setStatus(err.message === 'Server starting…' ? 'Just a moment…' : 'Something went wrong. Try again.', true);
      })
      .finally(function () {
        pending = false;
      });
  }

  function loop() {
    captureAndSend();
    animationId = requestAnimationFrame(loop);
  }

  function stop() {
    if (animationId != null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    revokePrevUrl();
    panelGrid.removeAttribute('src');
    startBtn.textContent = 'Tap to start';
    startBtn.disabled = false;
    setStatus('');
  }

  startBtn.addEventListener('click', function () {
    if (stream) {
      stop();
      return;
    }
    setStatus('Allow camera when your browser asks.');
    startBtn.disabled = true;

    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then(function (s) {
        stream = s;
        video.srcObject = s;
        video.onloadedmetadata = function () {
          video.play().then(function () {
            setStatus('Loading…');
            startBtn.textContent = 'Stop demo';
            startBtn.disabled = false;
            fpsTime = performance.now();
            frameCount = 0;
            loop();
          }).catch(function () {
            setStatus('Could not play video', true);
            startBtn.disabled = false;
          });
        };
      })
      .catch(function (err) {
        setStatus('Camera is needed for this demo. Allow it in your browser settings and try again.', true);
        startBtn.disabled = false;
      });
  });
})();
