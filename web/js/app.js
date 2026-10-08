// Entry point of the standalone bundle: every page is bundled into one script and the
// page to run is chosen by <body data-page="...">. Each page module only runs when loaded.
const pages = {
  index: () => import('./pages/index.js'),
  core: () => import('./pages/core.js'),
  market: () => import('./pages/market.js'),
  models: () => import('./pages/models.js'),
  instruments: () => import('./pages/instruments.js'),
  amc: () => import('./pages/amc.js'),
  exposure: () => import('./pages/exposure.js'),
  collateral: () => import('./pages/collateral.js'),
  allocation: () => import('./pages/allocation.js'),
  cva: () => import('./pages/cva.js'),
  wwr: () => import('./pages/wwr.js'),
  hedging: () => import('./pages/hedging.js'),
  gpu: () => import('./pages/gpu.js'),
};

const id = document.body.dataset.page || 'index';
(pages[id] || pages.index)();
