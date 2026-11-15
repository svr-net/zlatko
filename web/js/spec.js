// Default specification shared by all pages (market, models, portfolio, CSA, simulation).
// Pages edit a copy through the spec editor; the current spec is kept in localStorage so
// every page works on the same configuration.

export function defaultSpec() {
  return {
    domestic: { times: [0.5, 1, 2, 5, 10], rates: [0.03, 0.031, 0.033, 0.035, 0.037] },
    hw: { a: 0.04, sigma: 0.009 },
    assets: [{ name: 'EURUSD', spot: 1.1, vol: 0.1, carry: { times: [1, 10], rates: [0.02, 0.025] } }],
    credits: [
      {
        name: 'CPTY',
        recovery: 0.4,
        quotes: [
          { maturity: 1, spread: 0.007 },
          { maturity: 3, spread: 0.0095 },
          { maturity: 5, spread: 0.012 },
          { maturity: 7, spread: 0.013 },
          { maturity: 10, spread: 0.014 },
        ],
        cir: { kappa: 0.4, theta: 0.02, xi: 0.1, y0: 0.012, substeps: 4 },
      },
    ],
    // Factor order: [rates, assets..., credits...]
    correlation: [
      [1, 0.2, 0],
      [0.2, 1, 0],
      [0, 0, 1],
    ],
    sim: { numPaths: 2000, seed: 2024, antithetic: true },
    grid: { type: 'standard', horizon: 10 },
    trades: [
      { type: 'swap', id: 'IRS 10y payer', notional: 10e6, fixedRate: 0.0355, start: 0, tenor: 10, freq: 2, direction: 'payer' },
      { type: 'swap', id: 'IRS 5y receiver', notional: 6e6, fixedRate: 0.034, start: 0, tenor: 5, freq: 2, direction: 'receiver' },
      { type: 'forward', id: 'EURUSD fwd 3y', asset: 'EURUSD', notional: 5e6, strike: 1.12, maturity: 3 },
      { type: 'option', id: 'EURUSD call 2y (short)', asset: 'EURUSD', optionType: 'call', notional: -4e6, strike: 1.15, expiry: 2 },
    ],
    csa: { enabled: false, thresholdCounterparty: 250e3, thresholdOwn: 250e3, mta: 50e3, independentAmount: 0, mpr: 10 / 250 },
    own: { recovery: 0.4, quotes: [{ maturity: 5, spread: 0.006 }] },
    xva: { fundingSpreadBorrow: 0.01, fundingSpreadLend: 0.0, marginFundingSpread: 0.01, hurdleRate: 0.1, capitalRatio: 0.08, riskWeight: 1, alpha: 1.4 },
    pfeQuantile: 0.95,
    counterparty: 0,
  };
}

export const defaultBermudan = () => ({
  notional: 10e6, fixedRate: 0.036, start: 1, tenor: 9, freq: 1, direction: 'receiver', degree: 3,
});

const STORAGE_KEY = 'ccr-spec-v1';

export function loadSpec() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return activateNettingSet({ ...defaultSpec(), ...JSON.parse(raw) });
  } catch (_) { /* storage unavailable: fall back to defaults */ }
  return defaultSpec();
}

export function saveSpec(spec) {
  storeActive(spec);
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(spec)); } catch (_) { /* ignore */ }
}

export function resetSpec() {
  try { localStorage.removeItem(STORAGE_KEY); } catch (_) { /* ignore */ }
  return defaultSpec();
}

/** Factor labels in correlation order. */
export function factorNames(spec) {
  return ['rates', ...spec.assets.map((a) => a.name), ...spec.credits.map((c) => c.name)];
}

/** Resizes the correlation matrix after assets/credits are added or removed (keeps known entries). */
export function resizeCorrelation(spec, previousNames) {
  const names = factorNames(spec);
  const old = spec.correlation || [];
  const index = new Map((previousNames || names).map((n, i) => [n, i]));
  spec.correlation = names.map((a, i) =>
    names.map((b, j) => {
      if (i === j) return 1;
      const ia = index.get(a), ib = index.get(b);
      return ia !== undefined && ib !== undefined && old[ia] && old[ia][ib] !== undefined ? old[ia][ib] : 0;
    }));
  return spec;
}

// ------------------------------------------------------------------ portfolio (book)
//
// spec.book = { name, active, nettingSets: [{ id, counterparty (credit name), csa, trades }] }.
// The active netting set is mirrored in spec.trades / spec.csa / spec.counterparty, so every page
// analyses it unchanged; the portfolio page runs an analysis on all of them ({ scope: 'book' }).

const clone = (x) => JSON.parse(JSON.stringify(x));
const creditName = (spec, i) => spec.credits[i]?.name ?? spec.credits[0]?.name ?? '';

/** Copies spec.trades / csa / counterparty into the active netting set of the book. */
export function storeActive(spec) {
  const ns = spec.book?.nettingSets?.[spec.book.active ?? 0];
  if (!ns) return spec;
  ns.trades = clone(spec.trades);
  ns.csa = clone(spec.csa);
  ns.counterparty = creditName(spec, spec.counterparty || 0);
  return spec;
}

/** Makes netting set `index` of the book the one the pages analyse (default: the stored active one). */
export function activateNettingSet(spec, index = spec.book?.active ?? 0) {
  const ns = spec.book?.nettingSets?.[index];
  if (!ns) return spec;
  spec.book.active = index;
  spec.trades = clone(ns.trades);
  spec.csa = clone(ns.csa || { ...defaultSpec().csa, enabled: false });
  const c = spec.credits.findIndex((x) => x.name === ns.counterparty);
  spec.counterparty = c >= 0 ? c : 0;
  return spec;
}

/** Switches the pages to another netting set: the current one is stored first. */
export function switchNettingSet(spec, index) {
  storeActive(spec);
  return activateNettingSet(spec, index);
}

/** A book holding the current netting set only (created on the first visit of the portfolio page). */
export function bookFromCurrent(spec, name = 'My portfolio') {
  return { name, active: 0, nettingSets: [{ id: 'Netting set 1', counterparty: creditName(spec, spec.counterparty || 0), csa: clone(spec.csa), trades: clone(spec.trades) }] };
}

/** Adds a counterparty (credit) with the given name and spread, and resizes the correlation matrix. */
export function addCounterparty(spec, name, spread = 0.015) {
  const prev = factorNames(spec);
  const base = spec.credits[0] ? clone(spec.credits[0]) : clone(defaultSpec().credits[0]);
  base.name = name;
  const ref = base.quotes[Math.floor(base.quotes.length / 2)]?.spread || spread;
  base.quotes = base.quotes.map((q) => ({ ...q, spread: q.spread * spread / ref }));
  base.cir = { ...base.cir, y0: spread };
  spec.credits.push(base);
  resizeCorrelation(spec, prev);
  return spec.credits.length - 1;
}

/**
 * An example book on the assets and credits of the spec: a rates desk and a structured desk with
 * the first counterparty, an FX desk (collateralised, bond collateral) with a second one, which is
 * added if the spec has only one.
 */
export function exampleBook(spec) {
  if (spec.credits.length < 2) addCounterparty(spec, 'CPTY B', 0.02);
  const [a, b] = [spec.credits[0].name, spec.credits[1].name];
  const asset = spec.assets[0]?.name;
  const csa = { enabled: true, thresholdCounterparty: 250e3, thresholdOwn: 250e3, mta: 50e3, independentAmount: 0, mpr: 10 / 250 };
  const sets = [
    { id: 'Rates desk', counterparty: a, csa: { ...csa, enabled: false }, trades: [
      { type: 'swap', id: 'IRS 10y payer', notional: 10e6, fixedRate: 0.0355, start: 0, tenor: 10, freq: 2, direction: 'payer' },
      { type: 'swap', id: 'IRS 5y receiver', notional: 6e6, fixedRate: 0.034, start: 0, tenor: 5, freq: 2, direction: 'receiver' },
    ] },
    asset && { id: 'FX desk', counterparty: b, csa: { ...csa, collateral: 'bond', haircut: 0.02, bondMaturity: 10 }, trades: [
      { type: 'forward', id: `${asset} fwd 3y`, asset, notional: 5e6, strike: spec.assets[0].spot * 1.02, maturity: 3 },
      { type: 'option', id: `${asset} call 2y (short)`, asset, optionType: 'call', notional: -4e6, strike: spec.assets[0].spot * 1.05, expiry: 2 },
    ] },
    { id: 'Structured', counterparty: b, csa, trades: [
      { type: 'bermudan', id: 'Bermudan 1y×5y receiver', notional: 5e6, fixedRate: 0.036, start: 1, tenor: 5, freq: 1, direction: 'receiver', settlement: 'physical', degree: 3 },
      { type: 'swap', id: 'IRS 7y payer', notional: 4e6, fixedRate: 0.035, start: 0, tenor: 7, freq: 2, direction: 'payer' },
    ] },
  ].filter(Boolean);
  return { name: 'Example book', active: 0, nettingSets: sets };
}

/** What a book refers to that the spec does not define: [message]. */
export function bookProblems(spec, book) {
  const assets = new Set(spec.assets.map((a) => a.name));
  const credits = new Set(spec.credits.map((c) => c.name));
  const out = [];
  if (!book?.nettingSets?.length) out.push('the portfolio has no netting sets');
  for (const ns of book?.nettingSets || []) {
    if (!credits.has(ns.counterparty)) out.push(`${ns.id}: unknown counterparty ${ns.counterparty}`);
    if (!ns.trades?.length) out.push(`${ns.id}: no trades`);
    for (const t of ns.trades || []) if (t.asset && !assets.has(t.asset)) out.push(`${ns.id} / ${t.id}: unknown asset ${t.asset}`);
  }
  return out;
}

const BOOKS_KEY = 'ccr-books-v1';

/** Saved portfolios: { name: book }. */
export function savedBooks() {
  try { return JSON.parse(localStorage.getItem(BOOKS_KEY) || '{}'); } catch (_) { return {}; }
}

export function saveBook(book) {
  const all = savedBooks();
  all[book.name] = clone(book);
  try { localStorage.setItem(BOOKS_KEY, JSON.stringify(all)); } catch (_) { /* ignore */ }
}

export function deleteBook(name) {
  const all = savedBooks();
  delete all[name];
  try { localStorage.setItem(BOOKS_KEY, JSON.stringify(all)); } catch (_) { /* ignore */ }
}
