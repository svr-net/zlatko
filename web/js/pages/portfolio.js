import { barChart, fmt, lineChart } from '../charts.js';
import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import {
  activateNettingSet, bookFromCurrent, bookProblems, deleteBook, exampleBook, saveBook, saveSpec, savedBooks, storeActive, switchNettingSet,
} from '../spec.js';
import { card, el, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'portfolio',
  title: 'Portfolio: every analysis on all netting sets',
  context: 'ccr/app: spec.scope = "book" · common scenarios · gpu::FusedOptions::extraDates',
  description: 'A portfolio (book) holds several netting sets, each facing a counterparty with its own CSA and trades written on the assets of the specification. ' +
    'Every analysis runs on all of them in the fused kernels (WebGPU, else the CPU kernels) on <i>the same scenarios</i>: the netting sets share the reporting dates, the simulation dates and the seed, ' +
    'and each simulates only its own counterparty\'s intensity, so the rates, FX and discount paths are pathwise identical across the book. ' +
    'Exposures, CVA, DVA, FVA, MVA and KVA of different netting sets then add up exactly to the book; quantiles such as PFE do not add and are shown per netting set. ' +
    'The other pages analyse the active netting set, chosen here or in their specification panel.',
});

const ANALYSES = {
  exposure: 'Exposure & netting', collateral: 'Collateral (CSA variants)', allocation: 'Allocation to trades', cva: 'CVA / DVA',
  xva: 'XVA (FVA, MVA, KVA)', wrongWayRisk: 'Wrong-way risk', hedging: 'XVA hedging', instruments: 'Instruments on scenarios',
};
const PAGE_OF = { exposure: 'exposure', collateral: 'collateral', allocation: 'allocation', cva: 'cva', xva: 'xva', wrongWayRisk: 'wwr', hedging: 'hedging', instruments: 'instruments' };
const ANALYSIS_KEY = 'ccr-portfolio-analysis';
const storedAnalysis = () => { try { const a = localStorage.getItem(ANALYSIS_KEY); return ANALYSES[a] ? a : 'xva'; } catch (_) { return 'xva'; } };

const editor = specEditor(page, ['market', 'assets', 'credit', 'own', 'xva', 'correlation', 'simulation', 'portfolio', 'csa']);
if (!page.spec.book) { page.spec.book = bookFromCurrent(page.spec); saveSpec(page.spec); }

// ------------------------------------------------------------------ book editor

const bookBox = el('section', { class: 'card portfolio-book' });
page.main.insertBefore(bookBox, editor);
const message = el('span', { class: 'status' });

const download = (name, text) => {
  const a = el('a', { href: URL.createObjectURL(new Blob([text], { type: 'application/json' })), download: name });
  document.body.append(a);
  a.click();
  a.remove();
};

const SECTIONS = ['market', 'assets', 'credit', 'own', 'xva', 'correlation', 'simulation', 'portfolio', 'csa'];
let editorNode = editor;

function commit(note) {
  saveSpec(page.spec);
  // The specification panel shows the active netting set: rebuild it from the stored spec
  // (specEditor reloads page.spec), then the book table from the same object.
  const fresh = specEditor(page, SECTIONS, { open: editorNode.open });
  editorNode.replaceWith(fresh);
  editorNode = fresh;
  renderBook();
  if (note) message.textContent = note;
}

// Edits in the specification panel (trades, CSA, counterparty) show in the book table.
page.main.addEventListener('change', (e) => { if (e.target.closest('details.spec')) setTimeout(renderBook, 0); });
page.main.addEventListener('click', (e) => { if (e.target.closest('details.spec') && e.target.tagName === 'BUTTON') setTimeout(renderBook, 0); });

function useBook(book, note) {
  const spec = page.spec;
  spec.book = JSON.parse(JSON.stringify(book));
  activateNettingSet(spec, Math.min(book.active ?? 0, book.nettingSets.length - 1));
  const problems = bookProblems(spec, spec.book);
  commit(problems.length ? `Loaded with problems: ${problems.join('; ')}` : note);
}

function renderBook() {
  const spec = page.spec;
  if (!spec.book) { spec.book = bookFromCurrent(spec); saveSpec(spec); }  // e.g. after "Reset to defaults"
  storeActive(spec);
  const book = spec.book;
  bookBox.innerHTML = '';
  bookBox.append(el('h2', { text: 'Portfolio' }));

  const name = el('input', { type: 'text', value: book.name, 'aria-label': 'portfolio name' });
  name.addEventListener('change', () => { book.name = name.value || 'portfolio'; commit(); });
  const saved = savedBooks();
  const pick = el('select', { 'aria-label': 'saved portfolios' }, el('option', { value: '', text: Object.keys(saved).length ? 'saved portfolios…' : 'no saved portfolios' }),
    Object.keys(saved).sort().map((n) => el('option', { value: n, text: n })));
  const file = el('input', { type: 'file', accept: 'application/json,.json', style: 'display:none' });
  file.addEventListener('change', async () => {
    try {
      const data = JSON.parse(await file.files[0].text());
      const b = data.book || data;
      if (!Array.isArray(b.nettingSets)) throw new Error('no nettingSets in the file');
      useBook({ name: b.name || 'imported', active: 0, nettingSets: b.nettingSets }, `Imported ${b.nettingSets.length} netting sets`);
    } catch (e) { message.textContent = 'Import failed: ' + e.message; }
  });
  bookBox.append(el('div', { class: 'spec-actions' },
    el('label', { class: 'field' }, el('span', { text: 'name' }), name),
    el('button', { text: 'Save', onclick: () => { saveBook(book); commit(`Saved “${book.name}”`); } }),
    pick,
    el('button', { text: 'Load', onclick: () => { if (pick.value) useBook(saved[pick.value], `Loaded “${pick.value}”`); } }),
    el('button', { text: 'Delete saved', onclick: () => { if (pick.value) { deleteBook(pick.value); commit(`Deleted “${pick.value}”`); } } }),
    el('button', { text: 'Export JSON', onclick: () => download(`${book.name.replace(/[^\w.-]+/g, '_')}.json`, JSON.stringify({ book }, null, 2)) }),
    el('button', { text: 'Import JSON', onclick: () => file.click() }), file,
    el('button', { text: 'Example book', onclick: () => useBook(exampleBook(spec), 'Example book: three netting sets, two counterparties') }),
    el('button', { text: 'New (current netting set only)', onclick: () => useBook(bookFromCurrent(spec, 'New portfolio'), 'New portfolio') })));

  const credits = spec.credits.map((c) => c.name);
  const rows = book.nettingSets.map((ns, i) => {
    const active = i === (book.active ?? 0);
    const cp = el('select', { 'aria-label': `counterparty of ${ns.id}` }, credits.map((n) => el('option', { value: n, text: n, selected: n === ns.counterparty })));
    cp.addEventListener('change', () => { ns.counterparty = cp.value; if (active) activateNettingSet(spec, i); commit(); });
    const csa = ns.csa || { enabled: false };
    const on = el('input', { type: 'checkbox', checked: !!csa.enabled, 'aria-label': `CSA of ${ns.id}` });
    on.addEventListener('change', () => { ns.csa = { ...spec.csa, ...csa, enabled: on.checked }; if (active) activateNettingSet(spec, i); commit(); });
    const kind = el('select', { 'aria-label': `collateral of ${ns.id}` }, ['cash', 'bond', 'asset'].map((k) => el('option', { value: k, text: k, selected: k === (csa.collateral || 'cash') })));
    kind.addEventListener('change', () => {
      ns.csa = { ...csa, collateral: kind.value, haircut: kind.value === 'cash' ? 0 : kind.value === 'bond' ? 0.02 : 0.15, bondMaturity: Math.max(10, spec.grid.horizon) };
      if (active) activateNettingSet(spec, i);
      commit();
    });
    const types = {};
    for (const t of ns.trades) types[t.type] = (types[t.type] || 0) + 1;
    return [
      active ? el('b', { text: `▸ ${ns.id}` }) : ns.id, cp, on, csa.enabled ? kind : '–', csa.enabled && csa.collateral && csa.collateral !== 'cash' ? fmt.pct(csa.haircut || 0, 0) : '–',
      `${ns.trades.length} (${Object.entries(types).map(([k, n]) => `${n} ${k}`).join(', ')})`,
      el('span', {},
        active ? el('span', { class: 'note', text: 'editing below ' }) : el('button', { text: 'edit', onclick: () => { switchNettingSet(spec, i); commit(`Editing ${ns.id} in the specification panel`); editorNode.scrollIntoView?.(); } }),
        el('button', { text: 'duplicate', onclick: () => {
          let k = 2;
          while (book.nettingSets.some((x) => x.id === `${ns.id} (${k})`)) k++;
          book.nettingSets.splice(i + 1, 0, { ...JSON.parse(JSON.stringify(ns)), id: `${ns.id} (${k})` });
          commit();
        } }),
        book.nettingSets.length > 1 ? el('button', { text: 'remove', onclick: () => {
          const a = book.active ?? 0;
          book.nettingSets.splice(i, 1);
          activateNettingSet(spec, a > i ? a - 1 : a === i ? 0 : a);
          commit();
        } }) : null),
    ];
  });
  table(bookBox, ['netting set', 'counterparty', 'CSA', 'collateral', 'haircut', 'trades', ''], rows);
  bookBox.append(el('div', { class: 'spec-actions' },
    el('button', { text: '+ netting set', onclick: () => {
      let k = book.nettingSets.length + 1;
      while (book.nettingSets.some((x) => x.id === `Netting set ${k}`)) k++;
      book.nettingSets.push({ id: `Netting set ${k}`, counterparty: credits[0], csa: { ...spec.csa, enabled: false },
        trades: [{ type: 'swap', id: 'IRS 5y payer', notional: 5e6, fixedRate: 0.035, start: 0, tenor: 5, freq: 2, direction: 'payer' }] });
      switchNettingSet(spec, book.nettingSets.length - 1);
      commit(`Added Netting set ${k}: edit its trades in the specification panel`);
    } }),
    message));
  bookBox.append(el('p', { class: 'note', html: 'Trades are written on the assets of the specification (FX / equity assets, the domestic curve) and each netting set faces one of its counterparties; ' +
    'add assets or counterparties in the specification panel. The active netting set (▸) is the one the specification panel and the other pages edit and analyse.' }));
}
renderBook();

// ------------------------------------------------------------------ analyses

const select = el('select', { 'aria-label': 'analysis' }, Object.entries(ANALYSES).map(([k, v]) => el('option', { value: k, text: v, selected: k === storedAnalysis() })));
page.toolbar.insertBefore(el('label', { class: 'engine-select' }, 'Analysis ', select), page.status);

const cols = {
  exposure: [['value at 0', 'value0'], ['EEPE 1y', 'eepe1y'], ['lifetime EPE', 'epeLife'], ['peak PFE', 'maxPfe']],
  instruments: [['value at 0', 'value0'], ['EEPE 1y', 'eepe1y'], ['peak PFE', 'maxPfe']],
  collateral: [['EEPE no CSA', 'eepeNoCsa'], ['EEPE specified CSA', 'eepeCsa']],
  cva: [['CVA', 'cva'], ['DVA', 'dva'], ['CVA − DVA', 'bilateral'], ['pathwise CVA', 'pathwiseCva']],
  xva: [['CVA', 'cva'], ['DVA', 'dva'], ['FCA', 'fca'], ['FBA', 'fba'], ['MVA', 'mva'], ['KVA', 'kva'], ['total', 'total']],
  allocation: [['CVA', 'totalCva'], ['Σ standalone CVA', 'standaloneCvaSum'], ['Σ incremental CVA', 'incrementalCvaSum'], ['netting saving', 'nettingSaving', 'pct']],
  wrongWayRisk: [['independent CVA', 'independentCva'], ['pathwise CVA, lowest ρ', 'pathwiseCvaLow'], ['pathwise CVA, highest ρ', 'pathwiseCvaHigh']],
  hedging: [['CVA', 'cva'], ['CVA DV01', 'cvaDv01'], ['parallel CS01', 'parallelCs01']],
};

function render(analysis, r, spec) {
  const T = r.total;
  const sets = r.nettingSets;
  const sumOf = (key) => sets.reduce((s, x) => s + (x.summary[key] ?? 0), 0);
  const head = {
    exposure: () => [{ label: 'Book EEPE (1y)', value: fmt.num(T.profile.eepe1y), hint: 'of the summed EE' }, { label: 'Lifetime EPE', value: fmt.num(T.profile.epeLife) },
      { label: 'Book value at 0', value: fmt.num(T.profile.expectedValue[0]) }],
    instruments: () => [{ label: 'Book value at 0', value: fmt.num(T.profile.expectedValue[0]) }, { label: 'Trades', value: String(T.trades.length) }],
    collateral: () => T.variants?.slice(0, 2).map((v) => ({ label: `EEPE: ${v.name}`, value: fmt.num(v.profile.eepe1y), hint: fmt.pct(v.eepeVsNoCsa, 1) + ' of no CSA' })),
    cva: () => [{ label: 'Book CVA', value: fmt.num(T.cva) }, { label: 'DVA', value: fmt.num(T.dva) }, { label: 'CVA − DVA', value: fmt.num(T.bilateral) },
      { label: 'Pathwise CVA', value: fmt.num(T.pathwiseCva) }],
    xva: () => T.xva && [['Total XVA', 'total'], ['CVA', 'cva'], ['DVA', 'dva'], ['FVA', 'fva'], ['MVA', 'mva'], ['KVA', 'kva']].map(([l, k]) => ({ label: l, value: fmt.num(T.xva[k]) })),
    allocation: () => [{ label: 'Book CVA', value: fmt.num(T.totalCva) }, { label: 'Σ standalone CVA', value: fmt.num(T.standaloneCvaSum) },
      { label: 'Netting saving', value: fmt.pct(T.nettingSaving, 1), hint: 'of the standalone CVAs' }],
    wrongWayRisk: () => T.results?.map((x) => ({ label: `ρ = ${x.rho}`, value: fmt.num(x.pathwiseCva), hint: `× ${x.multiplier.toFixed(3)} vs independence` })),
    hedging: () => [{ label: 'Book CVA', value: fmt.num(T.cva) }, { label: 'CVA DV01', value: fmt.num(T.cvaDv01, 1) },
      ...(T.assetDeltas || []).map((d) => ({ label: `${d.name} CVA delta`, value: fmt.num(d.hedgeDelta), hint: `hedge ${fmt.compact(-d.hedgeUnits)} forward units` }))],
  }[analysis]();
  tiles(page.content, [...(head || []), { label: 'Netting sets', value: String(sets.length), hint: `${r.counterparties.length} counterparties · ${fmt.num(r.numPaths)} paths · ${r.simulationDates} dates` }]);

  const g = grid(page.content);
  const t = Array.from(r.times);
  const c = cols[analysis];
  const box = card(g, 'By netting set', `Each netting set on the common scenarios; the last row is the book.${analysis === 'exposure' || analysis === 'instruments' ? ' PFE does not add across netting sets.' : ''}`);
  const cell = (v, kind) => (kind === 'pct' ? fmt.pct(v, 1) : fmt.num(v));
  table(box, ['netting set', 'counterparty', 'trades', 'CSA', ...c.map((x) => x[0]), ''], [
    ...sets.map((x, i) => [x.id, x.counterparty, String(x.trades), x.collateralised ? 'yes' : '–', ...c.map(([, k, kind]) => cell(x.summary[k], kind)),
      el('a', { href: `${PAGE_OF[analysis]}.html`, text: 'open', onclick: () => { switchNettingSet(spec, i); saveSpec(spec); } })]),
    ...([[el('b', { text: 'book' }), '', String(sets.reduce((s, x) => s + x.trades, 0)), '', ...c.map(([, k, kind]) =>
      (kind === 'pct' ? fmt.pct(T.nettingSaving, 1) : k === 'maxPfe' ? '–' : k === 'eepe1y' ? fmt.num(T.profile.eepe1y) : k === 'epeLife' ? fmt.num(T.profile.epeLife)
        : k === 'value0' ? fmt.num(T.profile.expectedValue[0]) : fmt.num(sumOf(k)))), '']]),
  ]);

  const prof = (x) => x.result.profile || x.result.variants?.[1]?.profile;
  const discounted = ['cva', 'xva', 'wrongWayRisk', 'hedging'].includes(analysis);
  lineChart(card(g, discounted ? 'Discounted expected exposure EE*' : 'Expected exposure', 'The book (sum) and each netting set.'), {
    x: t, xLabel: 't (years)',
    series: [
      { name: 'book', y: Array.from(discounted ? T.profile.discountedEe : T.profile.ee), colorIndex: 0 },
      ...sets.filter((x) => prof(x)).map((x, i) => ({ name: x.id, y: Array.from(discounted ? prof(x).discountedEe : prof(x).ee), colorIndex: (i + 1) % 8, dash: true })),
    ],
  });
  if (T.byCounterparty) {
    const keys = Object.keys(T.byCounterparty[0]).filter((k) => !['name', 'nettingSets'].includes(k));
    const b = card(g, 'By counterparty', 'Netting sets with the same counterparty added up.');
    const label = (k) => ({ parallelCs01: 'parallel CS01', pathwiseCva: 'pathwise CVA' }[k] || k.toUpperCase());
    table(b, ['counterparty', 'netting sets', ...keys.map(label)], T.byCounterparty.map((x) => [x.name, Array.from(x.nettingSets).join(', '), ...keys.map((k) => fmt.num(x[k]))]));
    barChart(card(g, 'By counterparty', ''), { labels: T.byCounterparty.map((x) => x.name), series: keys.slice(0, 4).map((k, i) => ({ name: label(k), values: T.byCounterparty.map((x) => x[k]), colorIndex: i })) });
  }
  if (analysis === 'cva') {
    lineChart(card(g, 'Book CVA accumulation', ''), { x: t, xLabel: 't (years)', series: [{ name: 'cumulative CVA', y: Array.from(T.cumulativeCva) }] });
  } else if (analysis === 'xva') {
    lineChart(card(g, 'Book XVA accumulation', 'Cumulative CVA, FCA, MVA and KVA of the book.'), {
      x: t, xLabel: 't (years)',
      series: [['CVA', T.cumulativeCva], ['FCA', T.cumulativeFca], ['MVA', T.cumulativeMva], ['KVA', T.cumulativeKva]].map(([n, y], i) => ({ name: n, y: Array.from(y), colorIndex: i })),
    });
    lineChart(card(g, 'Book initial margin E[D·IM]', 'The initial margin of every netting set (nested conditional IM unless the proxy is chosen), added up.'), {
      x: t, xLabel: 't (years)', series: [{ name: 'E[D·IM]', y: Array.from(T.initialMargin) }],
    });
  } else if (analysis === 'collateral') {
    lineChart(card(g, 'Book EE by CSA variant', 'Every netting set under the same CSA variant, added up.'), {
      x: t, xLabel: 't (years)', series: T.variants.map((v, i) => ({ name: v.name, y: Array.from(v.profile.ee), colorIndex: i })),
    });
    table(card(g, 'CSA variants on the book', ''), ['variant', 'book EEPE 1y', 'vs no CSA'], T.variants.map((v) => [v.name, fmt.num(v.profile.eepe1y), fmt.pct(v.eepeVsNoCsa, 1)]));
  } else if (analysis === 'wrongWayRisk') {
    barChart(card(g, 'Book CVA against the correlation ρ', 'Pathwise CVA with the counterparty intensity correlated to the driver, against independence.'), {
      labels: T.results.map((x) => `ρ ${x.rho}`),
      series: [{ name: 'pathwise CVA', values: T.results.map((x) => x.pathwiseCva) }, { name: 'independent CVA', values: T.results.map((x) => x.independentCva), colorIndex: 2 }],
    });
  } else if (analysis === 'hedging') {
    table(card(g, 'Book hedges', 'CVA deltas add up across netting sets: one hedge per asset for the book.'), ['asset', 'CVA delta', 'hedge (forward units)'],
      T.assetDeltas.map((d) => [d.name, fmt.num(d.hedgeDelta), fmt.num(-d.hedgeUnits)]));
  }
  if (T.trades) {
    const tr = T.trades;
    const fields = { xva: [['CVA', 'cva'], ['FCA', 'fca'], ['KVA', 'kva']], allocation: [['marginal CVA', 'marginalCva'], ['incremental', 'incrementalCva'], ['standalone', 'standaloneCva']],
      instruments: [['type', 'type', 's'], ['maturity', 'maturity', 'y'], ['value at 0', 'value0'], ['benchmark', 'benchmark'], ['peak EE', 'eePeak']] }[analysis];
    const b = card(g, 'Trades of the book', analysis === 'instruments' ? '' : 'Euler allocation within each netting set: the contributions add up to the netting set, and the netting sets to the book.');
    table(b, ['trade', 'netting set', ...fields.map((f) => f[0])], tr.map((x) => [x.id, x.nettingSet, ...fields.map(([, k, kind]) =>
      (kind === 's' ? String(x[k]) : kind === 'y' ? fmt.years(x[k]) : fmt.num(x[k])))]));
    if (analysis !== 'instruments') barChart(card(g, 'Contribution by trade', ''), { labels: tr.map((x) => x.id), series: fields.map(([n, k], i) => ({ name: n, values: tr.map((x) => x[k]), colorIndex: i })) });
  }
}

const button = runButton(page, 'Run on portfolio', async (spec) => {
  const t0 = performance.now();
  const analysis = select.value;
  storeActive(spec);
  const problems = bookProblems(spec, spec.book);
  if (problems.length) throw new Error(problems.join('; '));
  const runResult = await runAnalysis(analysis, { ...spec, scope: 'book', samplePaths: 0 });
  const r = runResult.result;
  if (r.error) throw new Error(r.error);
  window.__ccrEngineRun = { page: 'portfolio', analysis, engine: runResult.engine, total: r.total, sets: r.nettingSets.map((x) => x.summary) };
  render(analysis, r, spec);
  return engineNote(runResult, performance.now() - t0);
});
select.addEventListener('change', () => { try { localStorage.setItem(ANALYSIS_KEY, select.value); } catch (_) { /* ignore */ } button.click(); });
engineSelector(page, () => button.click());
