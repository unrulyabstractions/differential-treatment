"use strict";
/* Results viewer: sidebar dataset switcher -> comparison tabs -> verdicts table
   (evidence bars, -log10 p, α tick) -> per-section plot cards with collapsible
   implicit/slices/conditional/calibration groups. Data: /api/index + each run's
   summary.json; files stream from /runs/. */

const state = { index: null, dataset: null, comparison: null, summaries: {} };
const $ = (sel, el = document) => el.querySelector(sel);

function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
function fmtStat(x) {
  if (x === null || x === undefined) return "–";
  const abs = Math.abs(x);
  if (abs !== 0 && (abs < 0.01 || abs >= 1000)) return x.toExponential(1);
  return String(+x.toPrecision(3));
}
function fmtP(p) {
  if (p === null || p === undefined) return "–";
  return p < 1e-4 ? "<1e-4" : String(+p.toPrecision(2));
}
function shortVariant(v) {
  return String(v)
    .replace("residual:google/gemma-2-2b-it", "gemma-2B")
    .replace("linear:residual:google/gemma-2-2b-it", "linear:gemma-2B")
    .replace("sentence-transformers/", "").replace("openai:", "").replace("cohere:", "");
}
function evidence(p) {
  if (p === null || p === undefined) return null;
  return Math.min(-Math.log10(Math.max(p, 1e-4)), 4);
}

const datasets = () => state.index.datasets;
const entryOf = (name) => datasets().find((d) => d.name === name);

async function boot() {
  try {
    state.index = await (await fetch("/api/index")).json();
  } catch (err) {
    $("#content").innerHTML = `<p class="chip warn">failed to load /api/index: ${esc(err)}</p>`;
    return;
  }
  $("#runs-root").textContent = state.index.runs_root;
  document.addEventListener("keydown", onKey);
  $("#lightbox").addEventListener("click", () => { $("#lightbox").hidden = true; });
  if (!datasets().length) {
    $("#content").innerHTML = `<p class="muted">no completed runs under ${esc(state.index.runs_root)}</p>`;
    return;
  }
  renderSidebar();
  applyHash();
  window.addEventListener("hashchange", applyHash);
}

/* deep links: #dataset or #dataset/comparison */
function applyHash() {
  const [name, comparison] = decodeURIComponent(location.hash.slice(1)).split("/");
  if (name && entryOf(name)) {
    selectDataset(name, comparison);
  } else {
    selectOverview();
  }
}
function syncHash() {
  const target = state.dataset ? `#${state.dataset}/${state.comparison || ""}` : "#";
  if (location.hash !== target) history.replaceState(null, "", target);
}

function renderSidebar() {
  const nav = $("#dataset-nav");
  nav.innerHTML = "";
  nav.appendChild(navItem("⌂ overview", state.dataset === null, selectOverview));
  for (const ds of datasets()) {
    const sig = ds.comparisons.reduce((a, c) => a + c.n_significant, 0);
    const total = ds.comparisons.reduce((a, c) => a + c.n_tests, 0);
    nav.appendChild(navItem(
      ds.name, state.dataset === ds.name,
      () => selectDataset(ds.name),
      `${sig}/${total} sig · ${ds.comparisons.length} comparisons`,
    ));
  }
}
function navItem(label, active, onClick, meta) {
  const button = document.createElement("button");
  button.className = "nav-item" + (active ? " active" : "");
  button.innerHTML = `<span>${esc(label)}</span>` + (meta ? `<small>${esc(meta)}</small>` : "");
  button.addEventListener("click", onClick);
  return button;
}

function selectOverview() {
  state.dataset = null;
  state.comparison = null;
  syncHash();
  renderSidebar();
  renderOverview();
  window.scrollTo(0, 0);
}
async function selectDataset(name, comparison) {
  state.dataset = name;
  const summary = await loadSummary(name);
  const names = summary.comparisons.map((c) => c.name);
  state.comparison = names.includes(comparison) ? comparison : names[0] || null;
  syncHash();
  renderSidebar();
  renderDataset();
  window.scrollTo(0, 0);
}
async function loadSummary(name) {
  if (!state.summaries[name]) {
    state.summaries[name] = await (await fetch(`/runs/${name}/summary.json`)).json();
  }
  return state.summaries[name];
}

/* expectation vs observed -> badge class (color never alone: text carries it) */
function badgeClass(comparison) {
  if (comparison.expectation === "null") return comparison.n_significant === 0 ? "ok" : "warn";
  if (comparison.expectation === "distinguishable") return comparison.n_significant > 0 ? "ok" : "warn";
  return "";
}

function renderOverview() {
  const cards = datasets().map((ds) => {
    const badges = ds.comparisons.map((c) =>
      `<span class="badge ${badgeClass(c)}" title="expected: ${esc(c.expectation || "n/a")}">` +
      `${esc(c.name)} · ${c.n_significant}/${c.n_tests}</span>`).join("");
    return `<article class="card ds-card" data-ds="${esc(ds.name)}">
      <h3>${esc(ds.name)}</h3>
      <p class="muted">${esc(ds.created_at)}</p>
      <div>${badges}</div>
    </article>`;
  }).join("");
  $("#content").innerHTML = `
    <header class="page-head">
      <h2>All datasets</h2>
      <p class="muted">gemma-2-2b-it residual as the sole representation · badge = tests significant
        (tinted when it matches the expectation, red-tinted when it violates it)</p>
    </header>
    <div class="grid overview-grid">${cards}</div>`;
  for (const el of document.querySelectorAll(".ds-card")) {
    el.addEventListener("click", () => selectDataset(el.dataset.ds));
  }
}

function renderDataset() {
  syncHash();
  const ds = entryOf(state.dataset);
  const summary = state.summaries[ds.name];
  const topPlots = ds.files.filter((f) => !f.includes("/") && f.endsWith(".png"));
  const tabs = summary.comparisons.map((c) =>
    `<button class="tab ${c.name === state.comparison ? "active" : ""}" data-comp="${esc(c.name)}">` +
    `<span>${esc(c.name)}</span><small>${c.n_significant}/${c.n_tests} sig · ${esc(c.expectation || "n/a")}</small></button>`).join("");
  const skipped = ds.skipped_variants.length
    ? `<p><span class="chip warn">skipped variants: ${esc(ds.skipped_variants.join(", "))}</span></p>` : "";
  $("#content").innerHTML = `
    <header class="page-head">
      <h2>${esc(ds.name)}</h2>
      <p class="muted">${esc(summary.created_at)} · dimensions: ${summary.dimensions_run.map(esc).join(", ")}</p>
      ${skipped}
    </header>
    ${topPlots.length ? `<div class="plot-row">${topPlots.map((f) => plotCard(ds.name, f)).join("")}</div>` : ""}
    <nav class="tabs">${tabs}</nav>
    <section id="comparison"></section>`;
  for (const el of document.querySelectorAll(".tab")) {
    el.addEventListener("click", () => { state.comparison = el.dataset.comp; renderDataset(); });
  }
  const comparison = summary.comparisons.find((c) => c.name === state.comparison) || summary.comparisons[0];
  if (comparison) renderComparison(ds, comparison);
  wireLightbox();
}

function renderComparison(ds, comparison) {
  const prefix = comparison.name + "/";
  const compFiles = ds.files.filter((f) => f.startsWith(prefix));
  const levelPlots = compFiles.filter((f) => !f.slice(prefix.length).includes("/") && f.endsWith(".png"));
  const sections = orderedSections(ds, compFiles, prefix);
  $("#comparison").innerHTML = `
    <div class="stat-strip">
      <span class="stat"><b>${comparison.n_significant}</b>/${comparison.n_tests} tests significant</span>
      <span class="stat">expected: <b>${esc(comparison.expectation || "n/a")}</b></span>
      <span class="stat">${esc(comparison.target.display_name)} (${comparison.target.n_prompts})
        vs ${esc(comparison.baseline.display_name)} (${comparison.baseline.n_prompts})</span>
    </div>
    ${levelPlots.length ? `<div class="plot-row">${levelPlots.map((f) => plotCard(ds.name, f)).join("")}</div>` : ""}
    <h3>Verdicts</h3>
    ${verdictTable(comparison)}
    ${sections.map((section) => sectionCard(ds, section)).join("")}`;
}

function orderedSections(ds, compFiles, prefix) {
  const names = [...new Set(
    compFiles.map((f) => f.slice(prefix.length).split("/")[0]).filter((s) => !s.includes(".")),
  )];
  const order = [...(state.summaries[ds.name].dimensions_run || []), "usage", "attributional"];
  const rank = (n) => { const i = order.indexOf(n); return i < 0 ? order.length : i; };
  names.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  return names.map((name) => ({
    name,
    prefix: prefix + name + "/",
    files: compFiles.filter((f) => f.startsWith(prefix + name + "/")),
  }));
}

function verdictTable(comparison) {
  const rows = comparison.verdicts.map((v) => {
    const ev = evidence(v.p_value);
    const sig = v.significant === null || v.significant === undefined ? "–" : v.significant ? "YES" : "no";
    const sigClass = sig === "YES" ? "sig-yes" : sig === "no" ? "sig-no" : "sig-na";
    const bar = ev === null ? "" :
      `<span class="ev-track"><span class="ev-fill ${v.significant ? "on" : "off"}" style="width:${(ev / 4) * 100}%"></span></span>`;
    return `<tr>
      <td>${esc(v.dimension)}</td>
      <td class="muted">${esc(v.test_name)}${v.variant ? " · " + esc(shortVariant(v.variant)) : ""}</td>
      <td class="num" title="${esc(v.statistic_name)}">${fmtStat(v.statistic_value)} <small class="muted">${esc(v.statistic_name)}</small></td>
      <td class="num">${fmtP(v.p_value)}</td>
      <td class="ev-cell" title="evidence −log₁₀ p · tick = α 0.05">${bar}</td>
      <td><span class="pill ${sigClass}">${sig}</span></td>
      <td class="detail" title="${esc(v.detail || "")}">${esc(v.detail || "")}</td>
    </tr>`;
  }).join("");
  return `<table class="verdicts">
    <thead><tr><th>section</th><th>test · variant</th><th>statistic</th><th>p</th>
    <th>evidence (−log₁₀ p)</th><th>sig</th><th>detail</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function sectionCard(ds, section) {
  const rel = (f) => f.slice(section.prefix.length);
  const top = section.files.filter((f) => !rel(f).includes("/"));
  const mainPlots = top.filter((f) => f.endsWith(".png"));
  const mainJsons = top.filter((f) => f.endsWith(".json"));
  const groups = {};
  for (const f of section.files.filter((f) => rel(f).includes("/"))) {
    const key = rel(f).split("/")[0];
    (groups[key] = groups[key] || []).push(f);
  }
  const groupHtml = Object.entries(groups).map(([name, files]) => {
    const plots = files.filter((f) => f.endsWith(".png"));
    const jsons = files.filter((f) => f.endsWith(".json"));
    return `<details class="group"><summary>${esc(name)} <small class="muted">· ${plots.length} plots, ${jsons.length} json</small></summary>
      <div class="grid plot-grid">${plots.map((f) => plotCard(ds.name, f, rel(f))).join("")}</div>
      <p>${jsonLinks(ds.name, jsons)}</p></details>`;
  }).join("");
  return `<article class="card section-card">
    <header><h3>${esc(section.name)}</h3>${jsonLinks(ds.name, mainJsons)}</header>
    <div class="grid plot-grid">${mainPlots.map((f) => plotCard(ds.name, f)).join("")}</div>
    ${groupHtml}</article>`;
}

function plotCard(dsName, file, caption) {
  const label = caption || file.split("/").pop().replace(/\.png$/, "");
  const src = `/runs/${dsName}/${file}`;
  return `<figure class="plot">
    <img loading="lazy" src="${esc(src)}" alt="${esc(label)}" data-full="${esc(src)}">
    <figcaption>${esc(label)}</figcaption></figure>`;
}
function jsonLinks(dsName, files) {
  if (!files.length) return "";
  return `<span class="json-links">${files.map((f) =>
    `<a href="/runs/${esc(dsName)}/${esc(f)}" target="_blank" rel="noopener">${esc(f.split("/").pop())}</a>`,
  ).join("")}</span>`;
}

function wireLightbox() {
  for (const img of document.querySelectorAll(".plot img")) {
    img.addEventListener("click", () => {
      $("#lightbox-img").src = img.dataset.full;
      $("#lightbox").hidden = false;
    });
  }
}

function onKey(event) {
  if (event.key === "Escape") { $("#lightbox").hidden = true; return; }
  if (!$("#lightbox").hidden) return;
  const names = datasets().map((d) => d.name);
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const i = names.indexOf(state.dataset);
    if (event.key === "ArrowDown") {
      selectDataset(state.dataset === null ? names[0] : names[Math.min(i + 1, names.length - 1)]);
    } else if (i <= 0) {
      selectOverview();
    } else {
      selectDataset(names[i - 1]);
    }
  }
  if ((event.key === "ArrowLeft" || event.key === "ArrowRight") && state.dataset) {
    const comps = state.summaries[state.dataset].comparisons.map((c) => c.name);
    const i = comps.indexOf(state.comparison);
    const j = event.key === "ArrowRight" ? Math.min(i + 1, comps.length - 1) : Math.max(i - 1, 0);
    if (j !== i) { state.comparison = comps[j]; renderDataset(); }
  }
}

boot();
