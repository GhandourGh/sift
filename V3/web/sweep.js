(function () {
  'use strict';

  var sweepBtn = document.getElementById('sweepBtn');
  var chartCanvas = document.getElementById('degradationChart');
  var placeholder = document.getElementById('chart-placeholder');
  var chartTabs = document.querySelectorAll('.chart-tab');

  var sweepData = null;  // Full sweep results
  var chart = null;       // Chart.js instance
  var activeTab = 'noise';

  // Algorithm colors (matching CSS)
  var COLORS = {
    edge: { line: '#00ff80', bg: 'rgba(0,255,128,0.1)' },
    sift: { line: '#ffb400', bg: 'rgba(255,180,0,0.1)' },
    cnn:  { line: '#b400ff', bg: 'rgba(180,0,255,0.1)' },
    yolo: { line: '#0080ff', bg: 'rgba(0,128,255,0.1)' },
  };

  var ALGO_LABELS = {
    edge: 'Edge Detection',
    sift: 'SIFT Features',
    cnn: 'CNN Classification',
    yolo: 'YOLO Detection',
  };

  // Perturbation labels for x-axis
  var PERTURB_LABELS = {
    noise: 'Noise Sigma',
    blur: 'Blur Kernel Size',
    rotation: 'Rotation Angle (deg)',
    brightness: 'Brightness Delta',
  };

  // --- Tab switching ---

  chartTabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      chartTabs.forEach(function (t) { t.classList.remove('active'); });
      tab.classList.add('active');
      activeTab = tab.getAttribute('data-type');
      if (sweepData && sweepData[activeTab]) {
        renderChart(activeTab);
      }
    });
  });

  // --- Sweep execution ---

  sweepBtn.addEventListener('click', function () {
    if (!window.cvDemo || !window.cvDemo.hasStream()) {
      window.cvDemo.setStatus('Start camera first', true);
      return;
    }

    var cvCanvas = window.cvDemo.canvas();
    if (!cvCanvas) {
      window.cvDemo.setStatus('No frame captured', true);
      return;
    }

    sweepBtn.disabled = true;
    sweepBtn.textContent = 'Running...';
    window.cvDemo.setStatus('Running sweep analysis (this takes ~10 seconds)...');

    cvCanvas.toBlob(function (blob) {
      if (!blob) {
        sweepBtn.disabled = false;
        sweepBtn.textContent = 'Run Sweep Analysis';
        return;
      }

      var formData = new FormData();
      formData.append('file', blob, 'sweep_frame.jpg');
      formData.append('steps', '10');

      fetch('/sweep-all', { method: 'POST', body: formData })
        .then(function (res) {
          if (!res.ok) throw new Error('Sweep failed: ' + res.status);
          return res.json();
        })
        .then(function (data) {
          sweepData = data;
          renderChart(activeTab);
          window.cvDemo.setStatus('Sweep complete! Click tabs to compare perturbation types.');
        })
        .catch(function (err) {
          window.cvDemo.setStatus('Sweep error: ' + err.message, true);
        })
        .finally(function () {
          sweepBtn.disabled = false;
          sweepBtn.textContent = 'Run Sweep Analysis';
        });
    }, 'image/jpeg', 0.9);
  });

  // --- Chart rendering ---

  function renderChart(perturbType) {
    if (!sweepData || !sweepData[perturbType]) return;

    var typeData = sweepData[perturbType];
    var steps = typeData.steps;

    // X-axis: perturbation strength values
    var labels = steps.map(function (s) {
      return Math.round(s.strength);
    });

    // Y-axis: normalized performance (0-100%) per algorithm
    var algos = ['edge', 'sift', 'cnn', 'yolo'];
    var datasets = algos.map(function (algo) {
      return {
        label: ALGO_LABELS[algo],
        data: steps.map(function (s) { return s.normalized[algo]; }),
        borderColor: COLORS[algo].line,
        backgroundColor: COLORS[algo].bg,
        borderWidth: 2.5,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.3,
        fill: false,
      };
    });

    // Show canvas, hide placeholder
    placeholder.style.display = 'none';
    chartCanvas.classList.add('visible');

    // Destroy old chart
    if (chart) {
      chart.destroy();
    }

    chart = new Chart(chartCanvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#e0e0e0',
              font: { size: 11 },
              usePointStyle: true,
              pointStyle: 'circle',
            },
          },
          tooltip: {
            backgroundColor: '#1e1e1e',
            borderColor: '#333',
            borderWidth: 1,
            titleColor: '#e0e0e0',
            bodyColor: '#ccc',
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%';
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: PERTURB_LABELS[perturbType] || 'Strength',
              color: '#888',
              font: { size: 11 },
            },
            ticks: { color: '#888', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.06)' },
          },
          y: {
            title: {
              display: true,
              text: 'Performance (% of baseline)',
              color: '#888',
              font: { size: 11 },
            },
            min: 0,
            max: 120,
            ticks: {
              color: '#888',
              font: { size: 10 },
              callback: function (val) { return val + '%'; },
            },
            grid: { color: 'rgba(255,255,255,0.06)' },
          },
        },
      },
    });
  }
})();
