// Shared page scaffolding: navigation, environment status, cards, tiles, tables,
// run buttons and the specification editor used by every page.
import { factorNames, loadSpec, resetSpec, resizeCorrelation, saveSpec } from './spec.js';
import { fmt } from './charts.js';

export const PAGES = [
  { group: 'Overview', items: [{ id: 'index', href: 'index.html', title: 'Overview' }] },
  {
    group: 'Foundations',
    items: [
      { id: 'core', href: 'core.html', title: 'Core numerics' },
      { id: 'market', href: 'market.html', title: 'Market data' },
      { id: 'models', href: 'models.html', title: 'Risk-factor models' },
    ],
  },
  {
    group: 'Pricing',
    items: [
      { id: 'instruments', href: 'instruments.html', title: 'Instruments on scenarios' },
      { id: 'amc', href: 'amc.html', title: 'American Monte Carlo' },
    ],
  },
  {
    group: 'Exposure',
    items: [
      { id: 'exposure', href: 'exposure.html', title: 'Exposure & netting' },
      { id: 'collateral', href: 'collateral.html', title: 'Collateral (CSA)' },
      { id: 'allocation', href: 'allocation.html', title: 'Exposure allocation' },
    ],
  },
  {
    group: 'Counterparty risk',
    items: [
      { id: 'cva', href: 'cva.html', title: 'CVA / DVA' },
      { id: 'wwr', href: 'wwr.html', title: 'Wrong-way risk' },
      { id: 'hedging', href: 'hedging.html', title: 'CVA hedging' },
    ],
  },
  { group: 'Acceleration', items: [{ id: 'gpu', href: 'gpu.html', title: 'WebGPU fused kernels' }] },
];

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

/** Builds the page shell. Returns { main, toolbar, status, content }. */
export function initPage({ id, title, context, description }) {
  document.title = `${title} · zlatko CCR`;
  // On narrow screens the links collapse behind a Menu button so the page content shows first.
  const links = el('div', { class: 'nav-links', id: 'nav-links' });
  const toggle = el('button', { class: 'menu-toggle', type: 'button', 'aria-expanded': 'false', 'aria-controls': 'nav-links', text: '☰ Menu' });
  const nav = el('nav', { class: 'sidebar' },
    el('div', { class: 'nav-head' },
      el('div', {}, el('div', { class: 'brand', text: 'zlatko · CCR' }),
        el('div', { class: 'sub', text: 'Counterparty credit exposure in WebAssembly + WebGPU' })),
      toggle),
    links);
  toggle.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
    toggle.textContent = open ? '✕ Close' : '☰ Menu';
  });
  for (const g of PAGES) {
    links.append(el('div', { class: 'group', text: g.group }));
    for (const p of g.items) links.append(el('a', { href: p.href, class: p.id === id ? 'active' : '', text: p.title }));
  }
  const wasmDot = el('span', { class: 'dot' });
  const gpuDot = el('span', { class: 'dot' });
  const wasmText = el('span', { text: 'WASM: loading…' });
  const gpuText = el('span', { text: 'WebGPU: checking…' });
  links.append(el('div', { class: 'env' }, el('div', {}, wasmDot, wasmText), el('div', {}, gpuDot, gpuText)));

  const status = el('span', { class: 'status', id: 'run-status' });
  const toolbar = el('div', { class: 'toolbar' });
  const content = el('div');
  const main = el('main', {},
    el('header', {}, el('div', { class: 'context', text: context }), el('h1', { text: title }), description ? el('p', { html: description }) : null),
    toolbar, content);
  toolbar.append(status);
  const app = document.getElementById('app');
  app.className = 'layout';
  app.append(nav, main);

  import('./ccr-client.js').then(({ run }) => run('version', {}))
    .then((v) => { wasmDot.className = 'dot ok'; wasmText.textContent = `WASM: ${v.library}`; })
    .catch((e) => { wasmDot.className = 'dot bad'; wasmText.textContent = 'WASM: ' + e.message; });
  gpuStatus().then((s) => { gpuDot.className = 'dot ' + (s.ok ? 'ok' : 'bad'); gpuText.textContent = 'WebGPU: ' + s.text; });

  return { main, toolbar, status, content, spec: loadSpec() };
}

export async function gpuStatus() {
  if (!('gpu' in navigator)) return { ok: false, text: 'not available' };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, text: 'no adapter' };
    const info = adapter.info || {};
    return { ok: true, text: [info.vendor, info.architecture].filter(Boolean).join(' ') || 'available' };
  } catch (e) {
    return { ok: false, text: e.message };
  }
}

export function card(parent, title, note) {
  const body = el('div');
  const c = el('section', { class: 'card' }, el('h2', { text: title }), note ? el('p', { class: 'note', html: note }) : null, body);
  parent.append(c);
  return body;
}

export function grid(parent, wide = false) {
  const g = el('div', { class: wide ? 'grid wide' : 'grid' });
  parent.append(g);
  return g;
}

export function tiles(parent, items) {
  const box = el('div', { class: 'tiles' });
  for (const it of items)
    box.append(el('div', { class: 'tile' },
      el('div', { class: 'label', text: it.label }), el('div', { class: 'value', text: it.value }),
      it.hint ? el('div', { class: 'hint', text: it.hint }) : null));
  parent.append(box);
  return box;
}

export function table(parent, headers, rows) {
  const t = el('table', { class: 'data' }, el('tr', {}, headers.map((h) => el('th', { text: h }))));
  for (const r of rows) t.append(el('tr', {}, r.map((c) => (c instanceof Node ? el('td', {}, c) : el('td', { text: c })))));
  const wrap = el('div', { class: 'table-wrap' }, t);
  parent.append(wrap);
  return wrap;
}

export const passFail = (ok, text) => el('span', { class: ok ? 'pass' : 'fail', text: (ok ? '✓ ' : '✗ ') + (text ?? (ok ? 'pass' : 'fail')) });

/** Adds a primary run button. handler(spec) may be async; status shows timing or errors. */
export function runButton(page, label, handler, { auto = true } = {}) {
  const btn = el('button', { class: 'primary', text: label });
  page.toolbar.insertBefore(btn, page.status);
  const go = async () => {
    btn.disabled = true;
    page.status.className = 'status';
    page.status.textContent = 'Running…';
    const t0 = performance.now();
    try {
      page.content.innerHTML = '';
      const msg = await handler(page.spec);
      page.status.textContent = msg || `Done in ${fmt.num(performance.now() - t0)} ms`;
    } catch (e) {
      console.error(e);
      page.status.className = 'status error';
      page.status.textContent = 'Error: ' + e.message;
    } finally {
      btn.disabled = false;
    }
  };
  btn.addEventListener('click', go);
  if (auto) setTimeout(go, 0);
  return btn;
}

// ------------------------------------------------------------------ spec editor

function numInput(get, set, { scale = 1, step = 'any', digits = 6 } = {}) {
  const v = get();
  const input = el('input', { type: 'number', step, value: v === undefined || v === null ? '' : +(v * scale).toFixed(digits) });
  input.addEventListener('change', () => set(input.value === '' ? null : Number(input.value) / scale));
  return input;
}

function textInput(get, set) {
  const input = el('input', { type: 'text', value: get() ?? '' });
  input.addEventListener('change', () => set(input.value));
  return input;
}

function selectInput(options, get, set) {
  const s = el('select', {}, options.map((o) => el('option', { value: o, text: o, selected: o === get() })));
  s.addEventListener('change', () => set(s.value));
  return s;
}

function field(label, input) { return el('label', { class: 'field' }, el('span', { text: label }), input); }

const TRADE_FIELDS = {
  swap: [['notional', 'notional'], ['fixedRate', 'fixed %', 100], ['start', 'start'], ['tenor', 'tenor'], ['freq', 'freq/yr'], ['direction', 'direction', ['payer', 'receiver']]],
  forward: [['asset', 'asset', 'assets'], ['notional', 'notional'], ['strike', 'strike'], ['maturity', 'maturity']],
  option: [['asset', 'asset', 'assets'], ['optionType', 'type', ['call', 'put']], ['notional', 'notional (−short)'], ['strike', 'strike'], ['expiry', 'expiry']],
  bermudan: [['notional', 'notional'], ['fixedRate', 'fixed %', 100], ['start', 'first exercise'], ['tenor', 'tenor'], ['freq', 'freq/yr'], ['direction', 'direction', ['payer', 'receiver']], ['settlement', 'settlement', ['physical', 'cash']], ['degree', 'regression degree']],
};

const TRADE_DEFAULTS = {
  swap: { notional: 5e6, fixedRate: 0.035, start: 0, tenor: 5, freq: 2, direction: 'payer' },
  forward: { notional: 1e6, strike: 1.1, maturity: 2 },
  option: { optionType: 'call', notional: 1e6, strike: 1.1, expiry: 1 },
  bermudan: { notional: 5e6, fixedRate: 0.036, start: 1, tenor: 5, freq: 1, direction: 'receiver', settlement: 'physical', degree: 3 },
};

/**
 * Collapsible editor for the shared specification.
 * sections: subset of ['market','assets','credit','correlation','simulation','portfolio','csa','own'].
 */
export function specEditor(page, sections, { open = false, tradeTypes = ['swap', 'forward', 'option'] } = {}) {
  page.spec = loadSpec();
  const details = el('details', { class: 'spec', open });
  const body = el('div', { class: 'spec-sections' });
  details.append(el('summary', { text: 'Specification (shared by all pages)' }), body);
  page.main.insertBefore(details, page.toolbar);
  const changed = () => saveSpec(page.spec);
  const render = () => {
    body.innerHTML = '';
    const spec = page.spec;
    const sec = (title) => { const s = el('div', { class: 'spec-section' }, el('h3', { text: title })); body.append(s); return s; };
    const rerender = () => { changed(); render(); };

    if (sections.includes('market')) {
      const s = sec('Domestic curve & Hull–White');
      const t = el('table', { class: 'edit' }, el('tr', {}, el('th', { text: 'tenor' }), el('th', { text: 'zero %' }), el('th')));
      spec.domestic.times.forEach((_, i) => t.append(el('tr', {},
        el('td', {}, numInput(() => spec.domestic.times[i], (v) => { spec.domestic.times[i] = v; changed(); })),
        el('td', {}, numInput(() => spec.domestic.rates[i], (v) => { spec.domestic.rates[i] = v; changed(); }, { scale: 100 })),
        el('td', {}, el('button', { text: '×', onclick: () => { spec.domestic.times.splice(i, 1); spec.domestic.rates.splice(i, 1); rerender(); } })))));
      s.append(t, el('button', { text: '+ pillar', onclick: () => { const n = spec.domestic.times.length; spec.domestic.times.push((spec.domestic.times[n - 1] || 0) + 5); spec.domestic.rates.push(spec.domestic.rates[n - 1] || 0.03); rerender(); } }));
      s.append(field('HW mean reversion a', numInput(() => spec.hw.a, (v) => { spec.hw.a = v; changed(); })));
      s.append(field('HW volatility σ (%)', numInput(() => spec.hw.sigma, (v) => { spec.hw.sigma = v; changed(); }, { scale: 100 })));
    }
    if (sections.includes('assets')) {
      const s = sec('FX / equity assets (log-normal)');
      const t = el('table', { class: 'edit' }, el('tr', {}, ['name', 'spot', 'vol %', 'carry %', ''].map((h) => el('th', { text: h }))));
      spec.assets.forEach((a, i) => {
        const carry = () => (a.carry.flat !== undefined ? a.carry.flat : a.carry.rates[0]);
        t.append(el('tr', {},
          el('td', {}, textInput(() => a.name, (v) => { const prev = factorNames(spec); spec.trades.forEach((tr) => { if (tr.asset === a.name) tr.asset = v; }); a.name = v; resizeCorrelation(spec, prev.map((n, k) => (k === 1 + i ? v : n))); rerender(); })),
          el('td', {}, numInput(() => a.spot, (v) => { a.spot = v; changed(); })),
          el('td', {}, numInput(() => a.vol, (v) => { a.vol = v; changed(); }, { scale: 100 })),
          el('td', {}, numInput(carry, (v) => { a.carry = { flat: v }; changed(); }, { scale: 100 })),
          el('td', {}, el('button', { text: '×', onclick: () => { const prev = factorNames(spec); spec.assets.splice(i, 1); resizeCorrelation(spec, prev); rerender(); } }))));
      });
      s.append(t, el('button', { text: '+ asset', onclick: () => { const prev = factorNames(spec); spec.assets.push({ name: 'ASSET' + (spec.assets.length + 1), spot: 100, vol: 0.2, carry: { flat: 0.01 } }); resizeCorrelation(spec, prev); rerender(); } }));
    }
    if (sections.includes('credit')) {
      const c = spec.credits[spec.counterparty || 0];
      const s = sec(`Counterparty credit (${c ? c.name : 'none'})`);
      if (c) {
        const t = el('table', { class: 'edit' }, el('tr', {}, ['CDS tenor', 'spread bp', ''].map((h) => el('th', { text: h }))));
        c.quotes.forEach((q, i) => t.append(el('tr', {},
          el('td', {}, numInput(() => q.maturity, (v) => { q.maturity = v; changed(); })),
          el('td', {}, numInput(() => q.spread, (v) => { q.spread = v; changed(); }, { scale: 1e4, digits: 3 })),
          el('td', {}, el('button', { text: '×', onclick: () => { c.quotes.splice(i, 1); rerender(); } })))));
        s.append(t, el('button', { text: '+ quote', onclick: () => { const last = c.quotes[c.quotes.length - 1]; c.quotes.push({ maturity: last.maturity + 5, spread: last.spread }); rerender(); } }));
        s.append(field('recovery %', numInput(() => c.recovery, (v) => { c.recovery = v; changed(); }, { scale: 100 })));
        s.append(field('CIR κ', numInput(() => c.cir.kappa, (v) => { c.cir.kappa = v; changed(); })));
        s.append(field('CIR θ (%)', numInput(() => c.cir.theta, (v) => { c.cir.theta = v; changed(); }, { scale: 100 })));
        s.append(field('CIR vol ξ', numInput(() => c.cir.xi, (v) => { c.cir.xi = v; changed(); })));
        s.append(field('CIR y0 (%)', numInput(() => c.cir.y0, (v) => { c.cir.y0 = v; changed(); }, { scale: 100 })));
        s.append(field('Euler sub-steps', numInput(() => c.cir.substeps, (v) => { c.cir.substeps = Math.max(1, Math.round(v)); changed(); }, { step: 1 })));
      }
    }
    if (sections.includes('own')) {
      const s = sec('Own credit (for DVA)');
      s.append(field('own 5y CDS (bp)', numInput(() => spec.own.quotes[0].spread, (v) => { spec.own.quotes[0].spread = v; changed(); }, { scale: 1e4, digits: 3 })));
      s.append(field('own recovery %', numInput(() => spec.own.recovery, (v) => { spec.own.recovery = v; changed(); }, { scale: 100 })));
    }
    if (sections.includes('correlation')) {
      const s = sec('Correlation of drivers');
      const names = factorNames(spec);
      const t = el('table', { class: 'edit' }, el('tr', {}, el('th'), names.map((n) => el('th', { text: n }))));
      names.forEach((n, i) => t.append(el('tr', {}, el('th', { text: n }), names.map((_, j) => el('td', {},
        i === j ? '1' : numInput(() => spec.correlation[i][j], (v) => { spec.correlation[i][j] = v; spec.correlation[j][i] = v; rerender(); }, { step: 0.05 }))))));
      s.append(t);
    }
    if (sections.includes('simulation')) {
      const s = sec('Simulation');
      s.append(field('paths', numInput(() => spec.sim.numPaths, (v) => { spec.sim.numPaths = Math.max(2, Math.round(v)); changed(); }, { step: 100 })));
      s.append(field('seed', numInput(() => spec.sim.seed, (v) => { spec.sim.seed = Math.round(v); changed(); }, { step: 1 })));
      const anti = el('input', { type: 'checkbox', checked: spec.sim.antithetic });
      anti.addEventListener('change', () => { spec.sim.antithetic = anti.checked; changed(); });
      s.append(el('label', { class: 'field check' }, el('span', { text: 'antithetic variates' }), anti));
      s.append(field('horizon (years)', numInput(() => spec.grid.horizon, (v) => { spec.grid.horizon = v; changed(); })));
      s.append(field('grid', selectInput(['standard', 'uniform'], () => spec.grid.type, (v) => { spec.grid.type = v; spec.grid.steps = spec.grid.steps || 40; changed(); })));
      s.append(field('PFE quantile %', numInput(() => spec.pfeQuantile, (v) => { spec.pfeQuantile = v; changed(); }, { scale: 100 })));
    }
    if (sections.includes('csa')) {
      const s = sec('Collateral agreement (CSA)');
      const on = el('input', { type: 'checkbox', checked: spec.csa.enabled });
      on.addEventListener('change', () => { spec.csa.enabled = on.checked; changed(); });
      s.append(el('label', { class: 'field check' }, el('span', { text: 'CSA enabled' }), on));
      s.append(field('threshold counterparty', numInput(() => spec.csa.thresholdCounterparty, (v) => { spec.csa.thresholdCounterparty = v; changed(); })));
      s.append(field('threshold own (−1 = one-way)', numInput(() => spec.csa.thresholdOwn, (v) => { spec.csa.thresholdOwn = v; changed(); })));
      s.append(field('minimum transfer amount', numInput(() => spec.csa.mta, (v) => { spec.csa.mta = v; changed(); })));
      s.append(field('independent amount', numInput(() => spec.csa.independentAmount, (v) => { spec.csa.independentAmount = v; changed(); })));
      s.append(field('margin period of risk (days)', numInput(() => spec.csa.mpr, (v) => { spec.csa.mpr = v; changed(); }, { scale: 250 })));
    }
    if (sections.includes('portfolio')) {
      const s = sec('Netting set (portfolio)');
      s.style.gridColumn = '1 / -1';
      spec.trades.forEach((tr, i) => {
        const row = el('div', { style: 'display:flex;flex-wrap:wrap;gap:6px;align-items:end;margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid var(--grid)' });
        const lab = (text, input) => el('label', { style: 'display:grid;font-size:11px;color:var(--text-secondary);min-width:90px' }, text, input);
        row.append(lab('type', selectInput(tradeTypes.includes(tr.type) ? tradeTypes : [...tradeTypes, tr.type], () => tr.type, (v) => { spec.trades[i] = { ...TRADE_DEFAULTS[v], type: v, id: tr.id, asset: spec.assets[0]?.name }; rerender(); })));
        row.append(lab('id', textInput(() => tr.id, (v) => { tr.id = v; changed(); })));
        for (const [key, label, opt] of TRADE_FIELDS[tr.type] || []) {
          let input;
          if (opt === 'assets') input = selectInput(spec.assets.map((a) => a.name), () => tr[key], (v) => { tr[key] = v; changed(); });
          else if (Array.isArray(opt)) input = selectInput(opt, () => tr[key], (v) => { tr[key] = v; changed(); });
          else input = numInput(() => tr[key], (v) => { tr[key] = v; changed(); }, { scale: opt || 1 });
          row.append(lab(label, input));
        }
        row.append(el('button', { text: 'remove', onclick: () => { spec.trades.splice(i, 1); rerender(); } }));
        s.append(row);
      });
      s.append(el('div', { class: 'spec-actions' }, tradeTypes.map((t) => el('button', {
        text: '+ ' + t,
        onclick: () => { spec.trades.push({ ...TRADE_DEFAULTS[t], type: t, id: `${t} ${spec.trades.length + 1}`, asset: spec.assets[0]?.name }); rerender(); },
      }))));
    }
    body.append(el('div', { class: 'spec-actions', style: 'grid-column:1/-1' },
      el('button', { text: 'Reset to defaults', onclick: () => { page.spec = resetSpec(); render(); } }),
      el('span', { class: 'status', text: 'Changes are saved automatically and shared across pages. Press Run to recompute.' })));
  };
  render();
  return details;
}

/** Selects a set of rows (sample paths) of numbers for display. */
export const toArrays = (rows) => Array.from(rows, (r) => Array.from(r));
