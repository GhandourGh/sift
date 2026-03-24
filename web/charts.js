(function () {
  'use strict';

  var chartsBtn = document.getElementById('chartsBtn');
  var chartsPanel = document.getElementById('chartsPanel');
  var chartsBackdrop = document.getElementById('chartsBackdrop');
  var chartsClose = document.getElementById('chartsClose');

  // ── Colors ──────────────────────────────────────────────────
  var COLORS = {
    edge: { solid: '#00ff80', bg: 'rgba(0,255,128,0.15)' },
    sift: { solid: '#ffb400', bg: 'rgba(255,180,0,0.15)' },
    cnn:  { solid: '#b400ff', bg: 'rgba(180,0,255,0.15)' },
    yolo: { solid: '#0080ff', bg: 'rgba(0,128,255,0.15)' },
  };

  var ALGO_NAMES = { edge: 'Edge', sift: 'SIFT', cnn: 'CNN', yolo: 'YOLO' };

  // ── History buffer ──────────────────────────────────────────
  var MAX_HISTORY = 60;
  var history = { edge: [], sift: [], cnn: [], yolo: [] };
  var metricHistory = { edge: [], sift: [], cnn: [], yolo: [] };
  var historyLabels = [];
  var frameIdx = 0;

  // ── Chart.js shared config ──────────────────────────────────
  var gridColor = 'rgba(255,255,255,0.05)';
  var tickColor = '#5c5c78';
  var tickFont = { size: 10, family: '-apple-system, system-ui, sans-serif' };

  Chart.defaults.color = '#e6e6f2';
  Chart.defaults.font.family = '-apple-system, system-ui, sans-serif';

  // ── 1. Processing Time Bar Chart ───────────────────────────
  var ctxTime = document.getElementById('chartTime').getContext('2d');
  var chartTime = new Chart(ctxTime, {
    type: 'bar',
    data: {
      labels: ['Edge', 'SIFT', 'CNN', 'YOLO'],
      datasets: [{
        label: 'Time (ms)',
        data: [0, 0, 0, 0],
        backgroundColor: [COLORS.edge.bg, COLORS.sift.bg, COLORS.cnn.bg, COLORS.yolo.bg],
        borderColor: [COLORS.edge.solid, COLORS.sift.solid, COLORS.cnn.solid, COLORS.yolo.solid],
        borderWidth: 2,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f0f18',
          borderColor: '#333',
          borderWidth: 1,
          callbacks: {
            label: function (ctx) { return ctx.parsed.y.toFixed(1) + ' ms'; },
          },
        },
      },
      scales: {
        x: { ticks: { color: tickColor, font: tickFont }, grid: { display: false } },
        y: {
          beginAtZero: true,
          title: { display: true, text: 'ms', color: tickColor, font: tickFont },
          ticks: { color: tickColor, font: tickFont },
          grid: { color: gridColor },
        },
      },
    },
  });

  // ── 2. Time History Line Chart ─────────────────────────────
  var ctxTimeHist = document.getElementById('chartTimeHistory').getContext('2d');
  var chartTimeHistory = new Chart(ctxTimeHist, {
    type: 'line',
    data: {
      labels: [],
      datasets: ['edge', 'sift', 'cnn', 'yolo'].map(function (algo) {
        return {
          label: ALGO_NAMES[algo],
          data: [],
          borderColor: COLORS[algo].solid,
          backgroundColor: COLORS[algo].bg,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35,
          fill: false,
        };
      }),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        legend: {
          position: 'top',
          labels: { usePointStyle: true, pointStyle: 'circle', boxWidth: 6, font: { size: 10 } },
        },
        tooltip: {
          mode: 'index', intersect: false,
          backgroundColor: '#0f0f18', borderColor: '#333', borderWidth: 1,
          callbacks: {
            label: function (ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + 'ms'; },
          },
        },
      },
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Frame', color: tickColor, font: tickFont },
          ticks: { color: tickColor, font: tickFont, maxTicksLimit: 10 },
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: 'ms', color: tickColor, font: tickFont },
          ticks: { color: tickColor, font: tickFont },
          grid: { color: gridColor },
        },
      },
    },
  });

  // ── 3. Algorithm Metrics Line Chart ────────────────────────
  var ctxMetrics = document.getElementById('chartMetrics').getContext('2d');
  var chartMetrics = new Chart(ctxMetrics, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Edge Density (%)',
          data: [],
          borderColor: COLORS.edge.solid,
          backgroundColor: COLORS.edge.bg,
          borderWidth: 2, pointRadius: 0, tension: 0.35, fill: true,
          yAxisID: 'y',
        },
        {
          label: 'SIFT Keypoints',
          data: [],
          borderColor: COLORS.sift.solid,
          backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.35,
          yAxisID: 'y1',
        },
        {
          label: 'CNN Confidence (%)',
          data: [],
          borderColor: COLORS.cnn.solid,
          backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.35,
          yAxisID: 'y',
        },
        {
          label: 'YOLO Person Conf (%)',
          data: [],
          borderColor: COLORS.yolo.solid,
          backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.35,
          yAxisID: 'y',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        legend: {
          position: 'top',
          labels: { usePointStyle: true, pointStyle: 'circle', boxWidth: 6, font: { size: 10 } },
        },
        tooltip: {
          mode: 'index', intersect: false,
          backgroundColor: '#0f0f18', borderColor: '#333', borderWidth: 1,
        },
      },
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Frame', color: tickColor, font: tickFont },
          ticks: { color: tickColor, font: tickFont, maxTicksLimit: 10 },
          grid: { display: false },
        },
        y: {
          position: 'left',
          beginAtZero: true,
          max: 100,
          title: { display: true, text: '% / Confidence', color: tickColor, font: tickFont },
          ticks: { color: tickColor, font: tickFont },
          grid: { color: gridColor },
        },
        y1: {
          position: 'right',
          beginAtZero: true,
          title: { display: true, text: 'Keypoints', color: tickColor, font: tickFont },
          ticks: { color: tickColor, font: tickFont },
          grid: { display: false },
        },
      },
    },
  });

  // ── Toggle panel ───────────────────────────────────────────
  function openCharts() {
    chartsPanel.classList.add('open');
    chartsBackdrop.classList.add('open');
    document.body.classList.add('charts-open');
    chartsBtn.textContent = 'Hide Charts';
    chartTime.resize();
    chartTimeHistory.resize();
    chartMetrics.resize();
  }

  function closeCharts() {
    chartsPanel.classList.remove('open');
    chartsBackdrop.classList.remove('open');
    document.body.classList.remove('charts-open');
    chartsBtn.textContent = 'Charts';
  }

  chartsBtn.addEventListener('click', function () {
    chartsPanel.classList.contains('open') ? closeCharts() : openCharts();
  });
  chartsClose.addEventListener('click', closeCharts);
  chartsBackdrop.addEventListener('click', closeCharts);

  // ── Feed data from app.js ──────────────────────────────────
  // app.js calls window.cvCharts.push(metrics) on each frame
  window.cvCharts = {
    push: function (metrics) {
      var e = metrics.edge || {};
      var s = metrics.sift || {};
      var c = metrics.cnn  || {};
      var y = metrics.yolo || {};

      frameIdx++;

      // --- Bar chart: latest times ---
      chartTime.data.datasets[0].data = [
        e.time_ms || 0,
        s.time_ms || 0,
        c.time_ms || 0,
        y.time_ms || 0,
      ];
      chartTime.update('none');

      // --- Time history ---
      history.edge.push(e.time_ms || 0);
      history.sift.push(s.time_ms || 0);
      history.cnn.push(c.time_ms || 0);
      history.yolo.push(y.time_ms || 0);
      historyLabels.push(frameIdx);

      if (historyLabels.length > MAX_HISTORY) {
        historyLabels.shift();
        history.edge.shift();
        history.sift.shift();
        history.cnn.shift();
        history.yolo.shift();
      }

      chartTimeHistory.data.labels = historyLabels;
      chartTimeHistory.data.datasets[0].data = history.edge;
      chartTimeHistory.data.datasets[1].data = history.sift;
      chartTimeHistory.data.datasets[2].data = history.cnn;
      chartTimeHistory.data.datasets[3].data = history.yolo;
      chartTimeHistory.update('none');

      // --- Metrics history ---
      metricHistory.edge.push(e.edge_density || 0);
      metricHistory.sift.push(s.keypoints_filtered || 0);
      metricHistory.cnn.push(c.top1_confidence || 0);
      metricHistory.yolo.push(y.person_confidence || 0);

      if (metricHistory.edge.length > MAX_HISTORY) {
        metricHistory.edge.shift();
        metricHistory.sift.shift();
        metricHistory.cnn.shift();
        metricHistory.yolo.shift();
      }

      chartMetrics.data.labels = historyLabels;
      chartMetrics.data.datasets[0].data = metricHistory.edge;
      chartMetrics.data.datasets[1].data = metricHistory.sift;
      chartMetrics.data.datasets[2].data = metricHistory.cnn;
      chartMetrics.data.datasets[3].data = metricHistory.yolo;
      chartMetrics.update('none');
    },

    reset: function () {
      frameIdx = 0;
      historyLabels.length = 0;
      history.edge.length = 0;
      history.sift.length = 0;
      history.cnn.length = 0;
      history.yolo.length = 0;
      metricHistory.edge.length = 0;
      metricHistory.sift.length = 0;
      metricHistory.cnn.length = 0;
      metricHistory.yolo.length = 0;
      chartTime.data.datasets[0].data = [0, 0, 0, 0];
      chartTime.update('none');
      chartTimeHistory.update('none');
      chartMetrics.update('none');
    },
  };
})();
