/* dtreat debug UI — vanilla JS single-page app over the /api routes.
   Charts are hand-rolled SVG per the mark specs: thin marks, rounded data
   ends, hairline grid, hover tooltips, legends for two series. */

"use strict";

const state = { runs: [], run: null, tab: "overview" };
const $ = (sel, el = document) => el.querySelector(sel);
const content = () => $("#content");

// ── palette from CSS custom properties (theme-correct at render time) ───
function cssVar(name) {
  return getComputedStyle($("#app")).getPropertyValue(name).trim();
}
const palette = () => ({
  target: cssVar("--series-target"),
  baseline: cssVar("--series-baseline"),
  seq: [cssVar("--seq-100"), cssVar("--seq-250"), cssVar("--seq-400"),
        cssVar("--seq-550"), cssVar("--seq-700")],
  ink: cssVar("--text-primary"),
  muted: cssVar("--text-muted"),
});
function communityColor(name, targetName) {
  return name === targetName ? palette().target : palette().baseline;
}

// ── tooltip ─────────────────────────────────────────────────────────────
const tooltip = $("#tooltip") || document.getElementById("tooltip");
function showTip(event, html) {
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  const x = Math.min(event.clientX + 14, window.innerWidth - 340);
  tooltip.style.left = x + "px";
  tooltip.style.top = event.clientY + 14 + "px";
}
function hideTip() { tooltip.style.display = "none"; }

// ── svg helpers ─────────────────────────────────────────────────────────
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}
function chartSvg(width, height) {
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%" });
  svg.style.maxWidth = width + "px";
  return svg;
}
function legendHtml(entries) {
  return `<div class="legend">` + entries.map(([label, color]) =>
    `<span><span class="swatch" style="background:${color}"></span>${esc(label)}</span>`
  ).join("") + `</div>`;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* Grouped horizontal bars: categories × 2 series (rates in [0,1]). */
function groupedBars(container, categories, series, opts = {}) {
  const rowH = 20, groupGap = 14, left = opts.left || 170, right = 60;
  const width = 720;
  const groupH = series.length * rowH + groupGap;
  const height = categories.length * groupH + 24;
  const svg = chartSvg(width, height);
  const xMax = opts.xMax || 1;
  const xw = width - left - right;
  // hairline grid at 0, .25, .5, .75, 1
  for (let i = 0; i <= 4; i++) {
    const x = left + (xw * i) / 4;
    svg.appendChild(svgEl("line", { x1: x, y1: 4, x2: x, y2: height - 20, class: "grid-line" }));
    const t = svgEl("text", { x, y: height - 6, "text-anchor": "middle", class: "axis-label" });
    t.textContent = (xMax * i / 4).toFixed(2);
    svg.appendChild(t);
  }
  categories.forEach((cat, ci) => {
    const y0 = ci * groupH + 6;
    const label = svgEl("text", {
      x: left - 8, y: y0 + (series.length * rowH) / 2 + 4,
      "text-anchor": "end", class: "axis-label",
    });
    const maxLabelChars = Math.max(24, Math.floor(left / 7));
    label.textContent = cat.length > maxLabelChars ? cat.slice(0, maxLabelChars - 1) + "…" : cat;
    svg.appendChild(label);
    series.forEach((s, si) => {
      const value = s.values[ci];
      if (value == null) return;
      const y = y0 + si * rowH;
      const w = Math.max(2, (value / xMax) * xw);
      const bar = svgEl("rect", {
        x: left, y, width: w, height: rowH - 6, rx: 4,
        fill: s.color, class: "mark",
      });
      bar.addEventListener("mousemove", e =>
        showTip(e, `<b>${esc(cat)}</b><br>${esc(s.name)}: ${value.toFixed(3)}${s.notes?.[ci] ? "<br>" + esc(s.notes[ci]) : ""}`));
      bar.addEventListener("mouseleave", hideTip);
      svg.appendChild(bar);
      const vl = svgEl("text", { x: left + w + 6, y: y + rowH - 9, class: "value-label" });
      vl.textContent = value.toFixed(2);
      svg.appendChild(vl);
    });
  });
  svg.appendChild(svgEl("line", { x1: left, y1: 4, x2: left, y2: height - 20, class: "baseline" }));
  container.insertAdjacentHTML("beforeend", legendHtml(series.map(s => [s.name, s.color])));
  container.appendChild(svg);
}

/* Diverging horizontal bars around zero (the Δ chart). Sign carries which
   community the behavior leans toward; non-significant bars are outlined. */
function divergingBars(container, items, opts = {}) {
  const rowH = 26, left = 170, right = 90, width = 720;
  const height = items.length * rowH + 30;
  const svg = chartSvg(width, height);
  const maxAbs = Math.max(0.05, ...items.map(d => Math.abs(d.value)));
  const xw = width - left - right;
  const x0 = left + xw / 2;
  const scale = (xw / 2 - 8) / maxAbs;
  for (const frac of [-1, -0.5, 0.5, 1]) {
    const x = x0 + frac * maxAbs * scale;
    svg.appendChild(svgEl("line", { x1: x, y1: 4, x2: x, y2: height - 22, class: "grid-line" }));
    const t = svgEl("text", { x, y: height - 8, "text-anchor": "middle", class: "axis-label" });
    t.textContent = (frac * maxAbs).toFixed(2);
    svg.appendChild(t);
  }
  items.forEach((d, i) => {
    const y = i * rowH + 6;
    const label = svgEl("text", { x: left - 8, y: y + rowH / 2 + 2, "text-anchor": "end", class: "axis-label" });
    label.textContent = d.label;
    svg.appendChild(label);
    const w = Math.abs(d.value) * scale;
    const color = d.value >= 0 ? opts.posColor : opts.negColor;
    const bar = svgEl("rect", {
      x: d.value >= 0 ? x0 : x0 - w, y, width: Math.max(2, w), height: rowH - 10, rx: 4,
      fill: d.significant ? color : "none",
      stroke: color, "stroke-width": d.significant ? 0 : 1.5,
      class: "mark",
    });
    bar.addEventListener("mousemove", e => showTip(e, d.tip));
    bar.addEventListener("mouseleave", hideTip);
    if (d.onClick) { bar.style.cursor = "pointer"; bar.addEventListener("click", d.onClick); }
    svg.appendChild(bar);
    // keep the value label clear of the category labels on long negative bars
    let labelX = d.value >= 0 ? x0 + w + 6 : x0 - w - 6;
    let anchor = d.value >= 0 ? "start" : "end";
    if (d.value < 0 && labelX < left + 30) { labelX = x0 + 6; anchor = "start"; }
    const vl = svgEl("text", {
      x: labelX, y: y + rowH / 2 + 3, "text-anchor": anchor, class: "value-label",
    });
    vl.textContent = (d.value >= 0 ? "+" : "") + d.value.toFixed(2) + (d.significant ? "" : " (ns)");
    svg.appendChild(vl);
  });
  svg.appendChild(svgEl("line", { x1: x0, y1: 4, x2: x0, y2: height - 22, class: "baseline" }));
  container.appendChild(svg);
}

/* Histogram: values → bins; optional vertical observed marker (ink, dashed). */
function histogram(container, values, opts = {}) {
  const width = 720, height = 180, left = 46, bottom = 24, top = 10, right = 16;
  const svg = chartSvg(width, height);
  if (!values.length) { container.appendChild(svg); return; }
  const lo = Math.min(...values, opts.marker ?? Infinity);
  const hi = Math.max(...values, opts.marker ?? -Infinity);
  const span = hi - lo || 1;
  const nBins = opts.bins || 30;
  const bins = new Array(nBins).fill(0);
  values.forEach(v => {
    const b = Math.min(nBins - 1, Math.floor(((v - lo) / span) * nBins));
    bins[b]++;
  });
  const maxCount = Math.max(...bins);
  const xw = width - left - right, yh = height - top - bottom;
  const barW = xw / nBins;
  bins.forEach((count, i) => {
    if (!count) return;
    const h = (count / maxCount) * yh;
    const binLo = lo + (span * i) / nBins;
    const bar = svgEl("rect", {
      x: left + i * barW + 1, y: top + yh - h,
      width: Math.max(1, barW - 2), height: h, rx: 2,
      fill: opts.color, class: "mark",
    });
    bar.addEventListener("mousemove", e =>
      showTip(e, `[${binLo.toFixed(3)}, ${(binLo + span / nBins).toFixed(3)}): <b>${count}</b>`));
    bar.addEventListener("mouseleave", hideTip);
    svg.appendChild(bar);
  });
  svg.appendChild(svgEl("line", { x1: left, y1: top + yh, x2: width - right, y2: top + yh, class: "baseline" }));
  [lo, lo + span / 2, hi].forEach((v, i) => {
    const t = svgEl("text", {
      x: left + (i * xw) / 2, y: height - 6, "text-anchor": "middle", class: "axis-label",
    });
    t.textContent = v.toFixed(2);
    svg.appendChild(t);
  });
  if (opts.marker != null) {
    const x = left + ((opts.marker - lo) / span) * xw;
    svg.appendChild(svgEl("line", {
      x1: x, y1: top, x2: x, y2: top + yh,
      stroke: palette().ink, "stroke-width": 2, "stroke-dasharray": "5 3",
    }));
    const t = svgEl("text", { x: x + 5, y: top + 12, class: "value-label" });
    t.textContent = `observed ${opts.marker.toFixed(3)}`;
    svg.appendChild(t);
  }
  container.appendChild(svg);
}

/* Heatmap: rows × cols with values in [0,1] on the sequential blue ramp. */
function heatmap(container, rowLabels, colLabels, valueAt, opts = {}) {
  const cell = 26, left = 150, top = 90;
  // generous right padding so rotated column labels don't clip
  const width = left + colLabels.length * cell + 140;
  const height = top + rowLabels.length * cell + 10;
  const svg = chartSvg(width, height);
  const seq = palette().seq;
  const colorFor = v => seq[Math.min(seq.length - 1, Math.floor(v * seq.length))];
  colLabels.forEach((c, j) => {
    const t = svgEl("text", {
      x: left + j * cell + cell / 2, y: top - 8, class: "axis-label",
      transform: `rotate(-40 ${left + j * cell + cell / 2} ${top - 8})`,
    });
    t.textContent = c;
    svg.appendChild(t);
  });
  rowLabels.forEach((r, i) => {
    const t = svgEl("text", { x: left - 8, y: top + i * cell + cell / 2 + 4, "text-anchor": "end", class: "axis-label" });
    t.textContent = r.length > 20 ? r.slice(0, 19) + "…" : r;
    svg.appendChild(t);
    colLabels.forEach((c, j) => {
      const v = valueAt(i, j);
      const rect = svgEl("rect", {
        x: left + j * cell + 1, y: top + i * cell + 1,
        width: cell - 2, height: cell - 2, rx: 3,
        fill: v == null ? "none" : colorFor(v),
        stroke: v == null ? cssVar("--grid") : "none",
        class: "mark",
      });
      rect.addEventListener("mousemove", e =>
        showTip(e, `<b>${esc(r)}</b> × ${esc(c)}<br>${v == null ? "no verdicts" : "rate " + v.toFixed(2)}${opts.rowNote ? "<br>" + esc(opts.rowNote(i)) : ""}`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
    });
  });
  container.appendChild(svg);
}

// ── data + rendering ────────────────────────────────────────────────────
async function api(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} on ${path}`);
  }
  return response.json();
}
function card(title, desc = "") {
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = `<h2>${esc(title)}</h2>` + (desc ? `<p class="desc">${esc(desc)}</p>` : "");
  content().appendChild(el);
  return el;
}
function tiles(items) {
  const row = document.createElement("div");
  row.className = "tile-row";
  row.innerHTML = items.map(t =>
    `<div class="tile ${t.cls || ""}"><div class="value">${t.value}</div><div class="label">${esc(t.label)}</div></div>`
  ).join("");
  content().appendChild(row);
}
function fail(error) {
  content().innerHTML = `<div class="card"><p class="muted">${esc(error.message)}</p></div>`;
}
function targetName(config) { return config?.target_community?.name || "target"; }
function baselineName(config) { return config?.baseline_community?.name || "baseline"; }

function syncHash() {
  location.hash = `run=${encodeURIComponent(state.run || "")}&tab=${state.tab}`;
}
function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.get("run")) state.run = params.get("run");
  if (params.get("tab") && renderers[params.get("tab")]) state.tab = params.get("tab");
}

async function loadRuns() {
  state.runs = await api("/api/runs");
  // auto-select: hash choice if valid, else most recently modified run
  if (!state.runs.some(r => r.run_name === state.run)) state.run = null;
  if (!state.run && state.runs.length) {
    state.run = [...state.runs].sort((a, b) => b.modified - a.modified)[0].run_name;
    render();
  }
  const list = $("#run-list");
  list.innerHTML = "";
  for (const run of state.runs) {
    const btn = document.createElement("button");
    btn.className = "run-item" + (state.run === run.run_name ? " active" : "");
    const dots = Object.values(run.stages)
      .map(done => `<span class="${done ? "done" : ""}">●</span>`).join("");
    btn.innerHTML = `<div>${esc(run.run_name)}</div><div class="stage-dots">${dots}</div>`;
    btn.onclick = () => { state.run = run.run_name; syncHash(); render(); loadRuns(); };
    list.appendChild(btn);
  }
  if (!state.runs.length) list.innerHTML = `<p class="muted">no runs under runs root yet</p>`;
}

const renderers = {
  async overview() {
    const data = await api(`/api/runs/${state.run}/overview`);
    const cfg = data.config;
    const headline = data.headline || {};
    tiles([
      { value: headline.significant_axes != null ? `${headline.significant_axes}/${headline.total_axes}` : "—", label: "significant axes (FDR)" },
      { value: headline.d_pi_bits != null ? headline.d_pi_bits.toFixed(2) + " bits" : "—", label: "D_π (significant axes)" },
      { value: headline.c2st_accuracy != null ? headline.c2st_accuracy.toFixed(3) : "—", label: "C2ST held-out accuracy" },
      { value: data.validation.problems.length, label: "validation problems", cls: data.validation.problems.length ? "bad" : "good" },
    ]);
    const c = card("Run configuration");
    c.innerHTML += `<pre class="raw mono">${esc(JSON.stringify(cfg, null, 2))}</pre>`;
    const v = card("Validation", "cross-stage consistency (dtreat validate)");
    const lines = [
      ...data.validation.problems.map(p => `<span class="chip warn">problem</span> ${esc(p)}`),
      ...data.validation.warnings.map(w => `<span class="chip">warn</span> ${esc(w)}`),
    ];
    v.innerHTML += lines.length ? lines.map(l => `<p>${l}</p>`).join("") : `<p class="muted">all checks passed</p>`;
  },

  async stage1() {
    const data = await api(`/api/runs/${state.run}/stage1`);
    const comp = data.comparability;
    tiles([
      { value: data.target_set.prompts.length, label: `${data.target_set.community} prompts` },
      { value: data.baseline_set.prompts.length, label: `${data.baseline_set.community} prompts` },
      { value: comp.total_variation_distance.toFixed(3), label: `TV distance (max ${comp.max_allowed_tv_distance})`, cls: comp.passed ? "good" : "bad" },
      { value: comp.chi2_p_value.toFixed(3), label: "χ² p-value (independence)" },
    ]);
    const chart = card("Instruction distribution", "Eq 2–3: both sets must ask the same things at the same rate");
    const instructions = comp.frequencies.map(f => f.instruction_id);
    groupedBars(chart, instructions, [
      { name: data.target_set.community, color: palette().target, values: comp.frequencies.map(f => f.target_fraction) },
      { name: data.baseline_set.community, color: palette().baseline, values: comp.frequencies.map(f => f.baseline_fraction) },
    ]);
    if (data.input_distinguishability) {
      const input = data.input_distinguishability;
      const inputCard = card("Input distinguishability (distinguish/ bridge)",
        `${input.n_significant}/${input.n_tests} tests significant` +
        (input.best_c2st_accuracy != null ? ` · best prompt C2ST ${input.best_c2st_accuracy.toFixed(3)} (${input.best_c2st_variant})` : ""));
      inputCard.innerHTML += `<table class="data"><tr><th>dimension</th><th>variant</th><th>statistic</th><th class="num">value</th><th class="num">p</th><th>sig</th></tr>` +
        input.verdicts.map(v =>
          `<tr><td>${esc(v.dimension)}</td><td class="mono">${esc(v.variant)}</td><td>${esc(v.statistic_name)}</td><td class="num">${v.statistic_value.toFixed(3)}</td><td class="num">${v.p_value == null ? "—" : v.p_value.toFixed(4)}</td><td>${v.significant == null ? "—" : v.significant ? '<span class="chip sig">yes</span>' : "no"}</td></tr>`).join("") + `</table>` +
        (input.skipped_variants.length ? `<p class="muted">skipped: ${esc(input.skipped_variants.join(", "))}</p>` : "");
    }
    for (const [key, set] of [["target_set", data.target_set], ["baseline_set", data.baseline_set]]) {
      const t = card(`${set.community} prompts`);
      t.innerHTML += `<table class="data"><tr><th>id</th><th>instruction</th><th>text</th></tr>` +
        set.prompts.map(p => `<tr><td class="mono">${esc(p.prompt_id)}</td><td>${esc(p.instruction_id)}</td><td>${esc(p.text)}</td></tr>`).join("") + `</table>`;
    }
  },

  async stage2() {
    const data = await api(`/api/runs/${state.run}/stage2`);
    const c = card("Hypothesized axes", `helper model: ${data.helper_model}`);
    c.innerHTML += `<table class="data"><tr><th>axis</th><th>question</th><th>rationale</th><th>source</th></tr>` +
      data.axes.map(a => {
        const sources = (a.sources && a.sources.length ? a.sources : [a.source])
          .map(s => `<span class="chip">${esc(s)}</span>`).join(" ");
        return `<tr><td class="mono">${esc(a.axis_id)}</td><td>${esc(a.question)}</td><td>${esc(a.rationale)}</td><td>${sources}</td></tr>`;
      }).join("") + `</table>`;
    const raw = card("Raw helper reply");
    raw.innerHTML += `<details><summary>show</summary><pre class="raw">${esc(data.raw_helper_reply)}</pre></details>`;
  },

  async stage3() {
    const overview = await api(`/api/runs/${state.run}/overview`);
    const tName = targetName(overview.config), bName = baselineName(overview.config);
    const controlsCard = card("Responses", "sampled from the target LLM (Eq 4)");
    controlsCard.innerHTML += `
      <div class="controls">
        <select id="s3-community"><option value="">all communities</option>
          <option>${esc(tName)}</option><option>${esc(bName)}</option></select>
        <input id="s3-search" placeholder="search text…">
      </div><div id="s3-table"></div>`;
    const lengthsCard = card("Response length by community", "characters per response");
    const draw = async () => {
      const community = $("#s3-community").value;
      const search = $("#s3-search").value;
      const data = await api(`/api/runs/${state.run}/stage3?limit=40&community=${encodeURIComponent(community)}&search=${encodeURIComponent(search)}`);
      const m = data.manifest;
      $("#s3-table").innerHTML =
        `<p class="desc">${data.total_matching} matching · ${m.collected_responses}/${m.expected_responses} collected · ${m.refusals} refusals · $${(m.estimated_cost_usd || 0).toFixed(4)}</p>` +
        `<table class="data"><tr><th>id</th><th>community</th><th>text</th><th></th></tr>` +
        data.records.map(r => `<tr><td class="mono">${esc(r.response_id)}</td>
          <td><span class="chip ${r.community === tName ? "target" : "baseline"}">${esc(r.community)}</span></td>
          <td>${esc(r.text.slice(0, 220))}${r.text.length > 220 ? "…" : ""}</td>
          <td>${r.refused ? '<span class="chip warn">refused</span>' : ""}</td></tr>`).join("") + `</table>`;
      lengthsCard.querySelectorAll("svg, .legend").forEach(el => el.remove());
      const lengths = data.lengths_by_community;
      for (const [community2, values] of Object.entries(lengths)) {
        lengthsCard.insertAdjacentHTML("beforeend",
          legendHtml([[`${community2} (n=${values.length})`, communityColor(community2, tName)]]));
        histogram(lengthsCard, values, { color: communityColor(community2, tName), bins: 24 });
      }
    };
    $("#s3-community").onchange = draw;
    $("#s3-search").oninput = () => { clearTimeout(state._t); state._t = setTimeout(draw, 300); };
    await draw();
  },

  async stage4() {
    const overview = await api(`/api/runs/${state.run}/overview`);
    const tName = targetName(overview.config);
    const data = await api(`/api/runs/${state.run}/stage4?limit=30`);
    const m = data.manifest;
    tiles([
      { value: m.scored_responses ?? data.total, label: "scored responses" },
      { value: m.unparsed_verdicts ?? 0, label: "unparsed/tie verdicts", cls: m.unparsed_verdicts ? "bad" : "good" },
      { value: m.judge_calls ?? "—", label: `judge calls (${m.judge_mode || "?"})` },
      { value: "$" + (m.estimated_cost_usd || 0).toFixed(4), label: "judge cost" },
    ]);
    const rates = card("Per-axis behavior rates", "share of responses exhibiting each behavior, by community");
    const axes = data.axis_ids;
    const communities = [...new Set(Object.values(data.per_axis_rates).flatMap(o => Object.keys(o)))];
    groupedBars(rates, axes, communities.map(community => ({
      name: community, color: communityColor(community, tName),
      values: axes.map(a => data.per_axis_rates[a]?.[community] ?? null),
    })));
    const hm = card("Prompt × axis heatmap", "per-prompt mean verdicts (the β_x of Eq 6)");
    heatmap(hm,
      data.heatmap.map(r => r.prompt_id), axes,
      (i, j) => data.heatmap[i][axes[j]],
      { rowNote: i => `community: ${data.heatmap[i].community}` });
    if (data.calibration) {
      const cal = data.calibration;
      const calCard = card("Judge calibration", `panel: ${cal.judge_models.join(", ")}`);
      if (cal.pair_agreements.length) {
        calCard.innerHTML += `<table class="data"><tr><th>judge pair</th><th class="num">n</th><th class="num">raw agreement</th><th class="num">Cohen κ</th></tr>` +
          cal.pair_agreements.map(p =>
            `<tr><td>${esc(p.judge_a)} vs ${esc(p.judge_b)}</td><td class="num">${p.n_paired_verdicts}</td><td class="num">${p.raw_agreement.toFixed(3)}</td><td class="num">${p.kappa_overall == null ? "—" : p.kappa_overall.toFixed(3)}</td></tr>`).join("") + `</table>`;
      }
      if (cal.axis_panel_agreements.length) {
        calCard.innerHTML += `<p class="desc">Fleiss κ per axis: ` +
          cal.axis_panel_agreements.map(a => `${esc(a.axis_id)} ${a.fleiss == null ? "—" : a.fleiss.toFixed(2)}`).join(" · ") + `</p>`;
      }
      if (cal.consistency.length) {
        calCard.innerHTML += `<p class="desc">self-consistency flip rates: ` +
          cal.consistency.map(c => `${esc(c.judge_model)}: ${(c.flip_rate_overall * 100).toFixed(1)}%`).join(" · ") + `</p>`;
      }
      for (const note of cal.notes || []) calCard.innerHTML += `<p class="muted">${esc(note)}</p>`;
    }
    const judged = card("Judged responses");
    judged.innerHTML += `<table class="data"><tr><th>response</th><th>community</th>` +
      axes.map(a => `<th>${esc(a)}</th>`).join("") + `<th>raw</th></tr>` +
      data.records.map(r => `<tr><td class="mono">${esc(r.response_id)}</td>
        <td><span class="chip ${r.community === tName ? "target" : "baseline"}">${esc(r.community)}</span></td>` +
        axes.map(a => {
          const v = r.verdicts[a];
          return `<td>${v === undefined ? '<span class="chip warn">?</span>' : v ? "✓" : "·"}</td>`;
        }).join("") +
        `<td><details><summary>raw</summary><pre class="raw">${esc(Object.entries(r.raw_judge_replies || {}).map(([j, reply]) => j + ":\n" + reply).join("\n\n"))}</pre></details></td></tr>`).join("") +
      `</table>`;
  },

  async stage5() {
    const data = await api(`/api/runs/${state.run}/stage5`);
    const report = data.report;
    const significant = report.axes.filter(a => a.significant);
    tiles([
      { value: `${significant.length}/${report.axes.length}`, label: `significant axes (BH-FDR ${report.fdr_alpha})` },
      { value: report.d_pi_bits_significant_axes != null ? report.d_pi_bits_significant_axes.toFixed(2) + " bits" : "—", label: "D_π significant axes (Eq 12)" },
      { value: report.c2st ? report.c2st.accuracy.toFixed(3) : "—", label: `C2ST accuracy (majority ${report.c2st ? report.c2st.majority_baseline.toFixed(3) : "—"})`, cls: report.c2st?.above_chance ? "bad" : "good" },
      { value: report.refusals ? `${(report.refusals.target_rate * 100).toFixed(1)}% / ${(report.refusals.baseline_rate * 100).toFixed(1)}%` : "—", label: "refusal rate target / baseline" },
    ]);
    if (report.input_output) {
      const io = report.input_output;
      const ioCard = card("Input legibility vs output treatment",
        "how much of the prompts' community signal shows up in the model's behavior");
      const rows = [];
      if (io.input_c2st_accuracy != null)
        rows.push({ label: "input: prompt separability (C2ST)", value: io.input_c2st_accuracy });
      if (io.output_c2st_accuracy != null)
        rows.push({ label: "output: behavior separability (C2ST)", value: io.output_c2st_accuracy });
      groupedBars(ioCard, rows.map(r => r.label), [{
        name: "held-out accuracy (0.5 = chance)", color: palette().seq[3],
        values: rows.map(r => r.value),
      }], { left: 260 });
      ioCard.innerHTML += `<p class="desc">signal usage: ${io.signal_usage == null ? "n/a" : (io.signal_usage * 100).toFixed(0) + "%"} · input tests significant: ${io.input_n_significant}/${io.input_n_tests}</p>` +
        `<p>${esc(io.interpretation)}</p>`;
    }
    const deltaCard = card("Treatment gaps Δ per axis (Eq 10)",
      `bar toward ${report.target_community} (right/blue) or ${report.baseline_community} (left/green); outline = not significant; click a bar for its permutation null`);
    const nullHost = document.createElement("div");
    divergingBars(deltaCard, report.axes.map(a => ({
      label: a.axis_id, value: a.delta, significant: a.significant,
      tip: `<b>${esc(a.axis_id)}</b><br>${esc(a.question)}<br>ẑ ${report.target_community}: ${a.rate_target.toFixed(2)} · ẑ ${report.baseline_community}: ${a.rate_baseline.toFixed(2)}<br>Δ=${a.delta.toFixed(3)} p=${a.p_value.toFixed(4)} q=${a.q_value.toFixed(4)}<br>I=${a.info_bits.toFixed(3)} bits · click for null dist`,
      onClick: async () => {
        nullHost.innerHTML = `<p class="desc">loading permutation null for ${esc(a.axis_id)}…</p>`;
        try {
          const nd = await api(`/api/runs/${state.run}/permutation-null/${a.axis_id}`);
          nullHost.innerHTML = `<p class="desc">permutation null for <b>${esc(a.axis_id)}</b> (${nd.null_deltas.length} label permutations, p=${nd.p_value.toFixed(4)})</p>`;
          histogram(nullHost, nd.null_deltas, { color: palette().seq[1], marker: nd.observed_delta });
        } catch (error) { nullHost.innerHTML = `<p class="muted">${esc(error.message)}</p>`; }
      },
    })), { posColor: palette().target, negColor: palette().baseline });
    deltaCard.appendChild(nullHost);

    const miCard = card("Mutual information ranking (Eq 13)", "bits the behavior on each axis reveals about the community");
    groupedBars(miCard, report.axes.map(a => a.axis_id), [{
      name: "I_j (bits)", color: palette().seq[3],
      values: report.axes.map(a => a.info_bits),
    }], { xMax: Math.max(0.25, ...report.axes.map(a => a.info_bits)) });

    if (report.method_breakdown && report.method_breakdown.length) {
      const methodCard = card("Per hypothesis-generation method",
        "how each method's axes fared on the shared responses");
      methodCard.innerHTML += `<table class="data"><tr><th>method</th><th class="num">axes</th><th class="num">significant</th><th class="num">total I (bits)</th><th class="num">mean |Δ|</th></tr>` +
        [...report.method_breakdown].sort((a, b) => b.total_info_bits - a.total_info_bits).map(m =>
          `<tr><td>${esc(m.method)}</td><td class="num">${m.n_axes}</td><td class="num">${m.n_significant}</td><td class="num">${m.total_info_bits.toFixed(3)}</td><td class="num">${m.mean_abs_delta.toFixed(3)}</td></tr>`).join("") + `</table>`;
    }
    const table = card("Full axis table");
    table.innerHTML += `<table class="data">
      <tr><th>axis</th><th>question</th><th class="num">ẑ_t</th><th class="num">ẑ_b</th><th class="num">Δ</th><th class="num">p</th><th class="num">q</th><th>sig</th><th class="num">I bits</th></tr>` +
      report.axes.map(a => `<tr><td class="mono">${esc(a.axis_id)}</td><td>${esc(a.question)}</td>
        <td class="num">${a.rate_target.toFixed(2)}</td><td class="num">${a.rate_baseline.toFixed(2)}</td>
        <td class="num">${a.delta >= 0 ? "+" : ""}${a.delta.toFixed(2)}</td>
        <td class="num">${a.p_value.toFixed(4)}</td><td class="num">${a.q_value.toFixed(4)}</td>
        <td>${a.insufficient_data ? '<span class="chip">n/a</span>' : a.significant ? '<span class="chip sig">yes</span>' : "no"}</td>
        <td class="num">${a.info_bits.toFixed(3)}</td></tr>`).join("") + `</table>`;

    const md = card("Analysis summary (markdown artifact)");
    md.innerHTML += `<pre class="raw">${esc(data.summary_markdown)}</pre>`;
  },

  async trace() {
    const host = card("LLM call trace", "every call across all stages; filter and drill in");
    host.innerHTML += `
      <div class="controls">
        <input id="tr-grep" placeholder="grep…">
        <select id="tr-errors"><option value="">all calls</option><option value="1">errors + refusals only</option></select>
      </div><div id="tr-body"></div>`;
    const draw = async () => {
      const grep = $("#tr-grep").value, errors = $("#tr-errors").value;
      const data = await api(`/api/runs/${state.run}/trace?grep=${encodeURIComponent(grep)}&errors_only=${errors ? "true" : "false"}&limit=200`);
      $("#tr-body").innerHTML =
        `<table class="data"><tr><th>role (model)</th><th class="num">calls</th><th class="num">cached</th><th class="num">errors</th><th class="num">refused</th><th class="num">tokens in→out</th><th class="num">cost</th></tr>` +
        Object.entries(data.aggregates).map(([k, s]) =>
          `<tr><td>${esc(k)}</td><td class="num">${s.calls}</td><td class="num">${s.cached}</td><td class="num">${s.errors}</td><td class="num">${s.refused}</td><td class="num">${s.input_tokens}→${s.output_tokens}</td><td class="num">$${s.cost_usd.toFixed(4)}</td></tr>`).join("") +
        `</table><p class="desc">finish reasons: ${esc(JSON.stringify(data.finish_reasons))} · ${data.total} matching records (last 200 shown)</p>` +
        `<table class="data"><tr><th>role</th><th>model</th><th class="num">ms</th><th>status</th><th>preview</th></tr>` +
        data.records.map(r => `<tr><td>${esc(r.role_label)}</td><td class="mono">${esc(r.model)}</td>
          <td class="num">${r.latency_ms}</td>
          <td>${r.error ? `<span class="chip warn">error</span>` : r.refused ? `<span class="chip warn">refused</span>` : r.cached ? `<span class="chip">cached</span>` : "ok"}</td>
          <td class="mono">${esc((r.error || r.preview || "").slice(0, 90))}</td></tr>`).join("") + `</table>`;
    };
    $("#tr-grep").oninput = () => { clearTimeout(state._t2); state._t2 = setTimeout(draw, 300); };
    $("#tr-errors").onchange = draw;
    await draw();
  },
};

async function render() {
  if (!state.run) return;
  content().innerHTML = "";
  try { await renderers[state.tab](); } catch (error) { fail(error); }
}

function activateTabButton() {
  document.querySelectorAll("#tabs button").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === state.tab));
}
for (const btn of document.querySelectorAll("#tabs button")) {
  btn.onclick = () => {
    state.tab = btn.dataset.tab;
    activateTabButton();
    syncHash();
    render();
  };
}
readHash();
activateTabButton();
if (state.run) render();
loadRuns();
