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
    if (raw) return { ...defaultSpec(), ...JSON.parse(raw) };
  } catch (_) { /* storage unavailable: fall back to defaults */ }
  return defaultSpec();
}

export function saveSpec(spec) {
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
