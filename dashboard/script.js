/**
 * script.js — Customer Segmentation
 * Đọc events.csv → tính RFM → chuẩn hoá → phân cụm → trả kết quả về dashboard
 */

'use strict';

// ─── Cấu hình ─────────────────────────────────────────────────────────────────
const CSV_PATH = '../data/events.csv';

// ─── State toàn cục ───────────────────────────────────────────────────────────
window.SEG = {
  rawData: null,      // mảng record sau khi parse
  features: null,     // Float64Array [n × 5]: R,F,M,B,A (đã chuẩn hoá)
  pcaCoords: null,    // Float64Array [n × 2]
  labels: null,       // Int32Array [n]
  userIds: null,      // BigInt64Array [n]
  nClusters: 0,
  silhouette: 0,
};

// ─── 1. ĐỌC CSV (Papa Parse streaming) ────────────────────────────────────────
/**
 * @param {File|string} source — File object hoặc path string
 * @param {function} onProgress — callback({loaded, total, phase})
 * @returns {Promise<void>} — điền vào SEG.rawData
 */
  function loadCSV(source, onProgress) {
  return new Promise((resolve, reject) => {
    const isFile = source instanceof File;

    const purchases = new Map(); // user_id → {times[], prices[], brands Set}

    let total = isFile ? source.size : 0;
    let loaded = 0;
    let rowCount = 0;

    const config = {
      header: true,
      skipEmptyLines: true,
      worker: false,
      chunk(results, parser) {
        for (const row of results.data) {
          if (row.event_type !== 'purchase') continue;
          const uid = row.user_id;
          const price = parseFloat(row.price);
          if (!uid || isNaN(price) || price <= 0) continue;

          const t = new Date(row.event_time.replace(' UTC', '')).getTime();
          if (isNaN(t)) continue;

          if (!purchases.has(uid)) {
            purchases.set(uid, { times: [], prices: [], brands: new Set() });
          }
          const rec = purchases.get(uid);
          rec.times.push(t);
          rec.prices.push(price);
          if (row.brand) rec.brands.add(row.brand.trim());
          rowCount++;
        }

        // progress estimate
        if (results.meta && results.meta.cursor) loaded = results.meta.cursor;
        onProgress?.({ loaded, total, phase: 'parsing', rows: rowCount });
      },
      complete() {
        onProgress?.({ loaded: total, total, phase: 'done', rows: rowCount });
        window.SEG.rawData = purchases;
        resolve(purchases);
      },
      error(err) { reject(err); },
    };

    if (isFile) {
      Papa.parse(source, config);
    } else {
      Papa.parse(source, { ...config, download: true });
    }
  });
}

// ─── 2. TÍNH ĐẶC TRƯNG RFM ────────────────────────────────────────────────────
/**
 * Chuyển map purchases → mảng đặc trưng và userIds
 * Đặc trưng: [Recency, Frequency, Monetary, Brand_Diversity, Avg_Basket]
 * Áp log1p cho F và M để giảm skewness
 */
  function buildFeatures(purchases) {
  let snapshotMs = 0;
  for (const rec of purchases.values()) {
    const mx = Math.max(...rec.times);
    if (mx > snapshotMs) snapshotMs = mx;
  }

  const ids = [];
  const raw = []; // [R, F, M, B, A]

  for (const [uid, rec] of purchases.entries()) {
    const maxT = Math.max(...rec.times);
    const R = Math.round((snapshotMs - maxT) / 86400000);
    const F = rec.prices.length;
    const M = rec.prices.reduce((s, v) => s + v, 0);
    const B = rec.brands.size;
    const A = M / F;
    ids.push(uid);
    raw.push([R, F, M, B, A]);
  }

  const n = raw.length;

  // Log1p transform cho F, M, A (right-skewed)
  const logRaw = raw.map(([R, F, M, B, A]) => [R, Math.log1p(F), Math.log1p(M), B, Math.log1p(A)]);

  // StandardScaler
  const nFeat = 5;
  const means = new Array(nFeat).fill(0);
  const stds  = new Array(nFeat).fill(0);

  for (const row of logRaw) for (let j = 0; j < nFeat; j++) means[j] += row[j];
  for (let j = 0; j < nFeat; j++) means[j] /= n;
  for (const row of logRaw) for (let j = 0; j < nFeat; j++) stds[j] += (row[j] - means[j]) ** 2;
  for (let j = 0; j < nFeat; j++) stds[j] = Math.sqrt(stds[j] / n) || 1;

  const Xs = new Float64Array(n * nFeat);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < nFeat; j++) {
      Xs[i * nFeat + j] = (logRaw[i][j] - means[j]) / stds[j];
    }
  }

  // Simpel PCA 2D (power iteration)
  const pca = simplePCA2D(Xs, n, nFeat);

  window.SEG.features  = Xs;
  window.SEG.pcaCoords = pca;
  window.SEG.userIds   = ids;
  window.SEG.rawRFM    = raw; // [R,F,M,B,A] trước chuẩn hoá

  return { Xs, pca, ids, raw, n };
}

// ─── 3. PCA 2D đơn giản (power iteration) ─────────────────────────────────────
    function simplePCA2D(Xs, n, nFeat) {
  // Covariance matrix (nFeat × nFeat)
  const cov = new Float64Array(nFeat * nFeat);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < nFeat; a++) {
      for (let b = a; b < nFeat; b++) {
        const v = Xs[i * nFeat + a] * Xs[i * nFeat + b];
        cov[a * nFeat + b] += v;
        if (a !== b) cov[b * nFeat + a] += v;
      }
    }
  }
  for (let i = 0; i < nFeat * nFeat; i++) cov[i] /= n;

  // Power iteration → 2 eigenvectors
    function powerIter(mat, dim, deflate) {
    let v = new Float64Array(dim).fill(1 / Math.sqrt(dim));
    for (let iter = 0; iter < 60; iter++) {
      let nv = new Float64Array(dim);
      for (let a = 0; a < dim; a++) for (let b = 0; b < dim; b++) nv[a] += mat[a * dim + b] * v[b];
      // deflate
      if (deflate) {
        const dot = deflate.reduce((s, vj, j) => s + nv[j] * vj, 0);
        for (let j = 0; j < dim; j++) nv[j] -= dot * deflate[j];
      }
      const norm = Math.sqrt(nv.reduce((s, x) => s + x * x, 0)) || 1;
      v = nv.map(x => x / norm);
    }
    return v;
  }

  const pc1 = powerIter(cov, nFeat, null);
  const pc2 = powerIter(cov, nFeat, pc1);

  const coords = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    let s1 = 0, s2 = 0;
    for (let j = 0; j < nFeat; j++) {
      s1 += Xs[i * nFeat + j] * pc1[j];
      s2 += Xs[i * nFeat + j] * pc2[j];
    }
    coords[i * 2]     = s1;
    coords[i * 2 + 1] = s2;
  }
  return coords;
}

// ─── 4. K-MEANS ───────────────────────────────────────────────────────────────
      function kMeans(Xs, n, nFeat, k, maxIter = 30) {
  // Khởi tạo bằng KMeans++ (cải tiến so với random)
  const centroids = [];
  const firstIdx = Math.floor(Math.random() * n);
  centroids.push(Array.from({ length: nFeat }, (_, j) => Xs[firstIdx * nFeat + j]));

  for (let c = 1; c < k; c++) {
    const dists = new Float64Array(n);
    let total = 0;
    for (let i = 0; i < n; i++) {
      let minD = Infinity;
      for (const cent of centroids) {
        let d = 0;
        for (let j = 0; j < nFeat; j++) d += (Xs[i * nFeat + j] - cent[j]) ** 2;
        if (d < minD) minD = d;
      }
      dists[i] = minD;
      total += minD;
    }
    let r = Math.random() * total;
    for (let i = 0; i < n; i++) {
      r -= dists[i];
      if (r <= 0) { centroids.push(Array.from({ length: nFeat }, (_, j) => Xs[i * nFeat + j])); break; }
    }
    if (centroids.length < c + 1) centroids.push(Array.from({ length: nFeat }, (_, j) => Xs[0 * nFeat + j]));
  }

  const labels = new Int32Array(n);
  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    // Assign
    for (let i = 0; i < n; i++) {
      let best = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        let d = 0;
        for (let j = 0; j < nFeat; j++) d += (Xs[i * nFeat + j] - centroids[c][j]) ** 2;
        if (d < bestD) { bestD = d; best = c; }
      }
      if (labels[i] !== best) { labels[i] = best; changed = true; }
    }
    if (!changed) break;
    // Update centroids
    const sums = Array.from({ length: k }, () => new Float64Array(nFeat));
    const cnts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      cnts[c]++;
      for (let j = 0; j < nFeat; j++) sums[c][j] += Xs[i * nFeat + j];
    }
    for (let c = 0; c < k; c++) {
      if (cnts[c] === 0) continue;
      for (let j = 0; j < nFeat; j++) centroids[c][j] = sums[c][j] / cnts[c];
    }
  }
  return labels;
}

// ─── 5. GMM (EM đơn giản, diagonal covariance) ───────────────────────────────
      function gmm(Xs, n, nFeat, k, maxIter = 20) {
  // Khởi tạo từ KMeans
  const initLabels = kMeans(Xs, n, nFeat, k, 10);

  const means  = Array.from({ length: k }, () => new Float64Array(nFeat));
  const vars   = Array.from({ length: k }, () => new Float64Array(nFeat).fill(1));
  const pis    = new Float64Array(k).fill(1 / k);
  const resps  = Array.from({ length: n }, () => new Float64Array(k));

  // Khởi tạo means từ KMeans
  const cnts = new Int32Array(k);
  for (let i = 0; i < n; i++) {
    const c = initLabels[i]; cnts[c]++;
    for (let j = 0; j < nFeat; j++) means[c][j] += Xs[i * nFeat + j];
  }
  for (let c = 0; c < k; c++) {
    if (cnts[c] > 0) for (let j = 0; j < nFeat; j++) means[c][j] /= cnts[c];
    pis[c] = cnts[c] / n;
  }

  for (let iter = 0; iter < maxIter; iter++) {
    // E-step
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let c = 0; c < k; c++) {
        let logP = Math.log(pis[c] + 1e-10);
        for (let j = 0; j < nFeat; j++) {
          const diff = Xs[i * nFeat + j] - means[c][j];
          logP -= 0.5 * (diff * diff / (vars[c][j] + 1e-6) + Math.log(vars[c][j] + 1e-6));
        }
        resps[i][c] = Math.exp(Math.max(logP, -300));
        sum += resps[i][c];
      }
      if (sum > 0) for (let c = 0; c < k; c++) resps[i][c] /= sum;
    }
    // M-step
    for (let c = 0; c < k; c++) {
      let Nc = 0;
      for (let i = 0; i < n; i++) Nc += resps[i][c];
      pis[c] = Nc / n;
      for (let j = 0; j < nFeat; j++) {
        let mu = 0, vr = 0;
        for (let i = 0; i < n; i++) mu += resps[i][c] * Xs[i * nFeat + j];
        mu /= (Nc + 1e-10);
        for (let i = 0; i < n; i++) vr += resps[i][c] * (Xs[i * nFeat + j] - mu) ** 2;
        means[c][j] = mu;
        vars[c][j]  = vr / (Nc + 1e-10) + 0.01;
      }
    }
  }

  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let c = 1; c < k; c++) if (resps[i][c] > resps[i][best]) best = c;
    labels[i] = best;
  }
  return labels;
}

// ─── 6. HDBSCAN đơn giản (density grid approach) ────────────────────────────
      function hdbscan(Xs, n, nFeat, minPts = 15) {
  // Dùng PCA 2D để tính khoảng cách trong không gian 2D cho nhanh
  const coords = window.SEG.pcaCoords;

  // Tính core distance
  const coreDists = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dists = [];
    const xi = coords[i * 2], yi = coords[i * 2 + 1];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = xi - coords[j * 2], dy = yi - coords[j * 2 + 1];
      dists.push(dx * dx + dy * dy);
    }
    dists.sort((a, b) => a - b);
    coreDists[i] = Math.sqrt(dists[Math.min(minPts - 1, dists.length - 1)]);
  }

  // Union-Find
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const rank   = new Int32Array(n);
  function find(x) { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
  function union(a, b) {
    a = find(a); b = find(b);
    if (a === b) return;
    if (rank[a] < rank[b]) [a, b] = [b, a];
    parent[b] = a;
    if (rank[a] === rank[b]) rank[a]++;
  }

  // Build MST greedily — connect if mutual reachability dist < threshold
  const eps = (coreDists.reduce((s, v) => s + v, 0) / n) * 1.5;
  for (let i = 0; i < n; i++) {
    const xi = coords[i * 2], yi = coords[i * 2 + 1];
    for (let j = i + 1; j < n; j++) {
      const dx = xi - coords[j * 2], dy = yi - coords[j * 2 + 1];
      const mrd = Math.max(coreDists[i], coreDists[j], Math.sqrt(dx * dx + dy * dy));
      if (mrd < eps) union(i, j);
    }
  }

  // Label clusters — nhóm có < minPts thành noise (-1)
  const compSize = new Map();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    compSize.set(r, (compSize.get(r) || 0) + 1);
  }
  const compId = new Map();
  let cid = 0;
  for (const [root, size] of compSize.entries()) {
    if (size >= minPts) compId.set(root, cid++);
  }

  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const r = find(i);
    labels[i] = compId.has(r) ? compId.get(r) : -1;
  }
  return labels;
}

// ─── 7. SILHOUETTE SCORE (sample) ─────────────────────────────────────────────
      function silhouetteScore(Xs, n, nFeat, labels, sampleSize = 2000) {
  const validIdx = [];
  for (let i = 0; i < n; i++) if (labels[i] >= 0) validIdx.push(i);
  if (validIdx.length < 2) return 0;

  // Random sample
  const sample = sampleSize >= validIdx.length
    ? validIdx
    : validIdx.sort(() => Math.random() - 0.5).slice(0, sampleSize);

  let totalSil = 0;
  for (const i of sample) {
    const ci = labels[i];
    const intraDs = [], interMap = new Map();

    for (const j of sample) {
      if (i === j) continue;
      let d = 0;
      for (let f = 0; f < nFeat; f++) d += (Xs[i * nFeat + f] - Xs[j * nFeat + f]) ** 2;
      d = Math.sqrt(d);
      if (labels[j] === ci) intraDs.push(d);
      else {
        const cj = labels[j];
        if (!interMap.has(cj)) interMap.set(cj, []);
        interMap.get(cj).push(d);
      }
    }

    const a = intraDs.length ? intraDs.reduce((s, v) => s + v, 0) / intraDs.length : 0;
    let b = Infinity;
    for (const ds of interMap.values()) {
      const avg = ds.reduce((s, v) => s + v, 0) / ds.length;
      if (avg < b) b = avg;
    }
    if (b === Infinity) b = 0;
    totalSil += (b - a) / (Math.max(a, b) || 1);
  }
  return +(totalSil / sample.length).toFixed(4);
}

// ─── 8. ENTRY POINT ───────────────────────────────────────────────────────────
/**
 * Chạy toàn bộ pipeline: load → features → cluster
 * @param {File|string} source
 * @param {string} algo 'kmeans' | 'gmm' | 'hdbscan'
 * @param {number} k
 * @param {function} onProgress
 * @returns {Promise<object>} kết quả để dashboard render
 */
      async function runPipeline(source, algo, k, onProgress) {
  // 1. Load
  onProgress?.({ phase: 'loading', pct: 0 });
  const purchases = await loadCSV(source, (p) => {
    const pct = p.total ? Math.round(p.loaded / p.total * 40) : 0;
    onProgress?.({ phase: 'loading', pct, rows: p.rows });
  });

  // 2. Features
  onProgress?.({ phase: 'features', pct: 45 });
  const { Xs, pca, ids, raw, n } = buildFeatures(purchases);
  const nFeat = 5;

  // 3. Cluster
  onProgress?.({ phase: 'clustering', pct: 60 });
  let labels;
  if (algo === 'kmeans')  labels = kMeans(Xs, n, nFeat, k);
  else if (algo === 'gmm') labels = gmm(Xs, n, nFeat, k);
  else                     labels = hdbscan(Xs, n, nFeat, 15);

  // 4. Silhouette
  onProgress?.({ phase: 'evaluating', pct: 85 });
  const sil = silhouetteScore(Xs, n, nFeat, labels);

  // 5. Cluster stats
  const nC = new Set(labels).size - (labels.includes(-1) ? 1 : 0);
  const stats = computeStats(raw, labels, nC);

  window.SEG.labels    = labels;
  window.SEG.nClusters = nC;
  window.SEG.silhouette = sil;

  onProgress?.({ phase: 'done', pct: 100 });

  return { n, nC, sil, labels, pca, ids, raw, stats };
}

// ─── 9. THỐNG KÊ TỪNG CỤM ────────────────────────────────────────────────────
      function computeStats(raw, labels, nC) {
  const sums  = Array.from({ length: nC }, () => [0, 0, 0, 0, 0]);
  const cnts  = new Array(nC).fill(0);
  const total = labels.length;

  for (let i = 0; i < total; i++) {
    const c = labels[i];
    if (c < 0 || c >= nC) continue;
    cnts[c]++;
    for (let j = 0; j < 5; j++) sums[c][j] += raw[i][j];
  }

  return Array.from({ length: nC }, (_, c) => ({
    cluster: c,
    count: cnts[c],
    pct: +((cnts[c] / total) * 100).toFixed(1),
    R: cnts[c] ? +(sums[c][0] / cnts[c]).toFixed(0) : 0,
    F: cnts[c] ? +(sums[c][1] / cnts[c]).toFixed(1) : 0,
    M: cnts[c] ? +(sums[c][2] / cnts[c]).toFixed(0) : 0,
    B: cnts[c] ? +(sums[c][3] / cnts[c]).toFixed(1) : 0,
    A: cnts[c] ? +(sums[c][4] / cnts[c]).toFixed(0) : 0,
  }));
}

window.renderAlgorithmComparison = function () {
    const canvas = document.getElementById('algo-chart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Xóa biểu đồ cũ nếu đã tồn tại
    if (window.algorithmChart) {
        window.algorithmChart.destroy();
    }

    window.algorithmChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['K-Means', 'Gaussian Mixture', 'HDBSCAN'],
            datasets: [{
                label: 'Silhouette Score',
                data: [0.4113, 0.0903, 0.0877],
                backgroundColor: [
                    '#3b82f6',
                    '#10b981',
                    '#f59e0b'
                ],
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Silhouette: ' + context.raw.toFixed(4);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 0.5,
                    title: {
                        display: true,
                        text: 'Silhouette Score'
                    }
                }
            }
        }
    });
}


