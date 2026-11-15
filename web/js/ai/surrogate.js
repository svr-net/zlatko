// Machine-learning surrogates of the XVA engine and their back test (no DOM, runs in Node).
//
// A surrogate maps a market state x = (counterparty CDS multiplier, own CDS multiplier,
// parallel rate shift, EURUSD volatility, funding spread) to the six adjustments of the default
// netting set. It is trained on full Monte Carlo prices of a Latin-hypercube of states and judged on
// a daily market path priced by the same engine (web/js/ai/backtest-data.js, examples/xva_ai_dataset.cpp).
//
// Models: polynomial ridge regression (degree 1 = the "first-order" sensitivity approximation,
// degree 2) and a small tanh multilayer perceptron trained with Adam. Everything is deterministic.

export const TARGETS = ['cva', 'dva', 'fca', 'fba', 'mva', 'kva'];
export const SIGNS = [1, -1, 1, -1, 1, 1]; // total = cva - dva + fca - fba + mva + kva
export const totalOf = (y) => y.reduce((s, v, i) => s + SIGNS[i] * v, 0);

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function standardiser(rows) {
  const d = rows[0].length, n = rows.length;
  const mean = new Array(d).fill(0), sd = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j] / n;
  for (const r of rows) for (let j = 0; j < d; j++) sd[j] += (r[j] - mean[j]) ** 2 / n;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  return { mean, sd, apply: (r) => r.map((v, j) => (v - mean[j]) / sd[j]), invert: (r) => r.map((v, j) => v * sd[j] + mean[j]) };
}

/** Solves the symmetric positive definite system A x = b (Cholesky). */
function solveSpd(A, b) {
  const n = b.length;
  const L = A.map((r) => r.slice());
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = L[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j];
    }
  }
  const y = new Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
  return x;
}

function polyFeatures(z, degree) {
  const f = [1, ...z];
  if (degree >= 2) for (let i = 0; i < z.length; i++) for (let j = i; j < z.length; j++) f.push(z[i] * z[j]);
  return f;
}

/** Ridge regression on polynomial features of the standardised state. */
export function fitRidge(X, Y, { degree = 2, lambda = 1e-6 } = {}) {
  const sx = standardiser(X), sy = standardiser(Y);
  const F = X.map((x) => polyFeatures(sx.apply(x), degree));
  const p = F[0].length;
  const A = Array.from({ length: p }, () => new Array(p).fill(0));
  for (const f of F) for (let i = 0; i < p; i++) for (let j = 0; j <= i; j++) A[i][j] += f[i] * f[j];
  for (let i = 0; i < p; i++) { for (let j = 0; j < i; j++) A[j][i] = A[i][j]; if (i > 0) A[i][i] += lambda * X.length; }
  const Z = Y.map((y) => sy.apply(y));
  const coef = Y[0].map((_, k) => {
    const b = new Array(p).fill(0);
    F.forEach((f, r) => { for (let i = 0; i < p; i++) b[i] += f[i] * Z[r][k]; });
    return solveSpd(A, b);
  });
  return {
    kind: degree === 1 ? 'linear' : `poly${degree}`,
    parameters: coef.length * p,
    predict: (x) => {
      const f = polyFeatures(sx.apply(x), degree);
      return sy.invert(coef.map((c) => c.reduce((s, v, i) => s + v * f[i], 0)));
    },
  };
}

/** tanh multilayer perceptron (layers = [in, h, h, out]) trained with full-batch Adam. */
export function fitMlp(X, Y, { hidden = 16, epochs = 2500, lr = 0.01, seed = 1, weightDecay = 1e-6, warm } = {}) {
  const sx = warm ? warm.sx : standardiser(X), sy = warm ? warm.sy : standardiser(Y);
  const Xs = X.map((x) => sx.apply(x)), Ys = Y.map((y) => sy.apply(y));
  const sizes = [X[0].length, hidden, hidden, Y[0].length];
  let W, B;
  if (warm) { W = warm.W.map((m) => m.map((r) => r.slice())); B = warm.B.map((b) => b.slice()); } else {
    const rnd = mulberry32(seed);
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    W = []; B = [];
    for (let l = 0; l < 3; l++) {
      W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => gauss() * Math.sqrt(1 / sizes[l]))));
      B.push(new Array(sizes[l + 1]).fill(0));
    }
  }
  const forward = (x) => {
    const a = [x];
    for (let l = 0; l < 3; l++) {
      const prev = a[l];
      const out = W[l].map((row, i) => { let s = B[l][i]; for (let j = 0; j < row.length; j++) s += row[j] * prev[j]; return l < 2 ? Math.tanh(s) : s; });
      a.push(out);
    }
    return a;
  };
  const m = { W: W.map((l) => l.map((r) => r.map(() => 0))), B: B.map((b) => b.map(() => 0)) };
  const v = { W: W.map((l) => l.map((r) => r.map(() => 0))), B: B.map((b) => b.map(() => 0)) };
  const n = Xs.length;
  for (let ep = 1; ep <= epochs; ep++) {
    const gW = W.map((l) => l.map((r) => r.map(() => 0))), gB = B.map((b) => b.map(() => 0));
    for (let s = 0; s < n; s++) {
      const a = forward(Xs[s]);
      let delta = a[3].map((o, k) => (2 * (o - Ys[s][k])) / (n * Ys[s].length));
      for (let l = 2; l >= 0; l--) {
        const prev = a[l];
        for (let i = 0; i < delta.length; i++) { gB[l][i] += delta[i]; for (let j = 0; j < prev.length; j++) gW[l][i][j] += delta[i] * prev[j]; }
        if (l > 0) {
          const next = new Array(prev.length).fill(0);
          for (let j = 0; j < prev.length; j++) { let t = 0; for (let i = 0; i < delta.length; i++) t += W[l][i][j] * delta[i]; next[j] = t * (1 - prev[j] * prev[j]); }
          delta = next;
        }
      }
    }
    const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** ep, c2 = 1 - b2 ** ep;
    for (let l = 0; l < 3; l++) {
      for (let i = 0; i < W[l].length; i++) {
        for (let j = 0; j < W[l][i].length; j++) {
          const g = gW[l][i][j] + weightDecay * W[l][i][j];
          m.W[l][i][j] = b1 * m.W[l][i][j] + (1 - b1) * g; v.W[l][i][j] = b2 * v.W[l][i][j] + (1 - b2) * g * g;
          W[l][i][j] -= (lr * (m.W[l][i][j] / c1)) / (Math.sqrt(v.W[l][i][j] / c2) + 1e-8);
        }
        const g = gB[l][i];
        m.B[l][i] = b1 * m.B[l][i] + (1 - b1) * g; v.B[l][i] = b2 * v.B[l][i] + (1 - b2) * g * g;
        B[l][i] -= (lr * (m.B[l][i] / c1)) / (Math.sqrt(v.B[l][i] / c2) + 1e-8);
      }
    }
  }
  const parameters = W.reduce((s, l) => s + l.length * l[0].length + l.length, 0);
  return { kind: 'mlp', parameters, state: { W, B, sx, sy }, predict: (x) => sy.invert(forward(sx.apply(x))[3]) };
}

// ------------------------------------------------------------------ metrics

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

/** Error statistics of predictions `p` against truths `t` (arrays of numbers), as a fraction of `scale` (gross notional). */
export function errorStats(t, p, scale) {
  const e = t.map((v, i) => p[i] - v);
  const abs = e.map(Math.abs);
  const sorted = abs.slice().sort((a, b) => a - b);
  return {
    mae: mean(abs), rmse: Math.sqrt(mean(e.map((v) => v * v))), bias: mean(e), max: sorted[sorted.length - 1],
    p95: sorted[Math.min(sorted.length - 1, Math.floor(0.95 * sorted.length))],
    maeBp: (1e4 * mean(abs)) / scale, mape: mean(t.map((v, i) => abs[i] / Math.max(Math.abs(v), 1e-9))),
  };
}

/** Daily P&L explain: how much of the change of the true XVA the surrogate's change reproduces. */
export function pnlExplain(t, p) {
  const dt = [], dp = [];
  for (let i = 1; i < t.length; i++) { dt.push(t[i] - t[i - 1]); dp.push(p[i] - p[i - 1]); }
  const resid = dt.map((v, i) => v - dp[i]);
  const varT = mean(dt.map((v) => v * v)) - mean(dt) ** 2;
  const varR = mean(resid.map((v) => v * v)) - mean(resid) ** 2;
  const signHit = mean(dt.map((v, i) => (Math.sign(v) === Math.sign(dp[i]) || Math.abs(v) < 1 ? 1 : 0)));
  return { effectiveness: 1 - varR / varT, signHit, residualSd: Math.sqrt(varR), actualSd: Math.sqrt(varT) };
}

export const outsideBox = (x, low, high) => x.some((v, j) => v < low[j] || v > high[j]);

// ------------------------------------------------------------------ back test

export const MODELS = {
  linear: { label: 'Linear (first order)', fit: (X, Y) => fitRidge(X, Y, { degree: 1, lambda: 1e-8 }) },
  poly2: { label: 'Quadratic ridge', fit: (X, Y) => fitRidge(X, Y, { degree: 2, lambda: 1e-6 }) },
  mlp: { label: 'MLP 5-16-16-6', fit: (X, Y, o = {}) => fitMlp(X, Y, { seed: 7, ...o }) },
};

/**
 * Back test on the daily market path.
 * `static`: fitted once on the training window. `walkForward`: refitted every `refitEvery` days on
 * the training window plus every earlier day (whose full Monte Carlo is available overnight).
 * Baselines: the engine run only on every fifth day, carried forward in between.
 * Returns per-model predictions and metrics. `onProgress(msg)` is called between fits.
 */
export async function runBacktest(data, { models = ['linear', 'poly2', 'mlp'], refitEvery = 21, mlpEpochs = 2500, mlpRefitEpochs = 400, onProgress = () => {} } = {}) {
  const { meta } = data;
  const trX = data.train.x, trY = data.train.y.map((y) => y.slice(0, 6));
  const days = data.backtest.x.length;
  const X = data.backtest.x, Y = data.backtest.y.map((y) => y.slice(0, 6));
  const truthTotal = Y.map(totalOf);
  const outside = X.map((x) => outsideBox(x, meta.low, meta.high));
  const out = { days, truthTotal, outside, models: {}, baseline: {}, outsideDays: outside.filter(Boolean).length };
  const tick = () => new Promise((r) => setTimeout(r, 0));

  const score = (predY) => {
    const predTotal = predY.map(totalOf);
    const sel = (mask) => { const idx = predY.map((_, i) => i).filter((i) => mask(i)); return { t: idx.map((i) => truthTotal[i]), p: idx.map((i) => predTotal[i]) }; };
    const all = sel(() => true), inR = sel((i) => !outside[i]), outR = sel((i) => outside[i]);
    const comp = {};
    TARGETS.forEach((name, k) => { comp[name] = errorStats(Y.map((y) => y[k]), predY.map((y) => y[k]), meta.grossNotional); });
    return {
      predTotal, total: errorStats(all.t, all.p, meta.grossNotional), inRange: inR.t.length ? errorStats(inR.t, inR.p, meta.grossNotional) : null,
      outRange: outR.t.length ? errorStats(outR.t, outR.p, meta.grossNotional) : null, components: comp, pnl: pnlExplain(truthTotal, predTotal),
    };
  };

  // Baselines: the previous day's engine value (lag 1) and the weekly engine run carried forward.
  const lag1 = Y.map((_, i) => Y[Math.max(0, i - 1)]);
  const weekly = Y.map((_, i) => Y[i - (i % 5)]);
  out.baseline.lag1 = { label: 'Yesterday\'s engine value', ...score(lag1) };
  out.baseline.weekly = { label: 'Engine run weekly, carried forward', ...score(weekly) };

  for (const key of models) {
    const spec = MODELS[key];
    onProgress(`fitting ${spec.label}`);
    await tick();
    let t0 = performance.now();
    let model = spec.fit(trX, trY, key === 'mlp' ? { epochs: mlpEpochs } : {});
    const fitMs = performance.now() - t0;
    t0 = performance.now();
    const staticPred = X.map((x) => model.predict(x));
    const inferMs = (performance.now() - t0) / X.length;
    const rolling = [];
    let cur = model;
    for (let i = 0; i < days; i++) {
      if (i > 0 && i % refitEvery === 0) {
        onProgress(`${spec.label}: walk-forward refit at day ${i}`);
        await tick();
        const aX = trX.concat(X.slice(0, i)), aY = trY.concat(Y.slice(0, i));
        cur = key === 'mlp' ? fitMlp(aX, aY, { seed: 7, epochs: mlpRefitEpochs, lr: 0.003, warm: cur.state }) : spec.fit(aX, aY);
      }
      rolling.push(cur.predict(X[i]));
    }
    out.models[key] = { label: spec.label, parameters: model.parameters, fitMs, inferMicros: inferMs * 1000, static: score(staticPred), walkForward: score(rolling), staticPred, rollingPred: rolling };
  }
  out.truthY = Y;
  return out;
}
