/* ══════════════════════════════════════════════════════════════════════
   Crucible — interface

   Six stages, each its own tab. A stage unlocks when the one before it has
   produced what it needs, so the interface can never be in a state where a
   control is visible but meaningless.
   ══════════════════════════════════════════════════════════════════════ */

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const escapeHtml = (value) =>
  String(value).replace(/[&<>"]/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[character]));
const fixed = (value, places = 4) =>
  value === null || value === undefined ? "—" : Number(value).toFixed(places);
const signed = (value, places = 4) =>
  value === null || value === undefined ? "—"
    : (value >= 0 ? "+" : "−") + Math.abs(value).toFixed(places);

const MECHANISM_TEXT = {
  REASON: "was used to assign the label",
  CONSEQUENCE: "exists only because the outcome happened",
  TIMING: "is recorded after the prediction point",
};

/* ── shared state ───────────────────────────────────────────────────── */

const app = {
  file: null,           // the File the user chose
  dictFile: null,       // optional data dictionary File
  table: null,          // { columns, rows } parsed in the browser
  target: null,
  predictionPoint: "",
  model: null,
  apiKey: "",
  keyProvider: null,       // which vendor issued it, once recognized
  jobId: null,
  audit: null,          // verdicts, correlations, buckets
  decisions: {},        // { column: "drop" | "keep" }
  impact: null,
  demo: false,
  unlocked: new Set(["overview"]),
};

/* ── tab router ─────────────────────────────────────────────────────── */

const TAB_ORDER = ["overview", "data", "model", "detect", "review", "results"];
// Settings sits outside the sequence: it is reachable at any time and gates
// nothing, because the shared pool means a key is never required to proceed.
const ALWAYS_OPEN = ["settings"];
let currentTab = "overview";

function showTab(name) {
  if (!app.unlocked.has(name) && !ALWAYS_OPEN.includes(name)) return;
  // Settings is reachable at any moment, including before the model step has
  // ever been visited, and it reports the shared pool. Without this it sat on
  // its placeholder indefinitely for anyone who opened it first.
  if (name === "settings") loadModels();
  currentTab = name;
  [...TAB_ORDER, ...ALWAYS_OPEN].forEach((tab) =>
    $(`tab-${tab}`).classList.toggle("on", tab === name));
  paintNav();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function unlock(name) {
  app.unlocked.add(name);
  paintNav();
}

function paintNav() {
  $("steps-nav").querySelectorAll("button").forEach((button) => {
    const tab = button.dataset.tab;
    const reached = app.unlocked.has(tab) || ALWAYS_OPEN.includes(tab);
    button.disabled = !reached;
    button.classList.toggle("on", tab === currentTab);
    button.classList.toggle(
      "done", reached && TAB_ORDER.indexOf(tab) < TAB_ORDER.indexOf(currentTab));
  });
}

$("steps-nav").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (button && !button.disabled) showTab(button.dataset.tab);
});

function status(text, kind = "") {
  $("topbar-status").textContent = text;
  $("topbar-status").className = `topbar-status ${kind}`;
}

/* ── comma separated values, parsed properly ────────────────────────── */

/* Published tables are not reliably comma separated, and archive exports open
   with a block of `#` lines describing the columns. Read with the wrong
   delimiter, or without skipping that block, a table collapses to one column of
   joined text and every later stage processes it without complaint. Both are
   handled here, and the same rule runs server-side in `intake.load_table`. */

const truncate = (text, limit) =>
  text.length <= limit ? text : `${text.slice(0, limit - 1)}\u2026`;

const DELIMITERS = [",", "\t", ";", "|"];

function stripCommentPreamble(text) {
  // Only a leading run counts. A `#` further down the file is data.
  const lines = text.split("\n");
  let start = 0;
  while (start < lines.length) {
    const line = lines[start].trim();
    if (line === "" || line.startsWith("#")) start++;
    else break;
  }
  return { body: lines.slice(start).join("\n"), skipped: start };
}

function scanDelimited(text, delimiter) {
  const rows = [];
  let row = [], field = "", inQuotes = false;
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (inQuotes) {
      if (character === '"') {
        if (text[index + 1] === '"') { field += '"'; index++; }
        else inQuotes = false;
      } else field += character;
    } else if (character === '"') inQuotes = true;
    else if (character === delimiter) { row.push(field); field = ""; }
    else if (character === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else field += character;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function parseCsv(text) {
  // Strip a byte order mark and normalize line endings before scanning.
  text = text.replace(/^﻿/, "").replace(/\r\n?/g, "\n");
  const { body, skipped } = stripCommentPreamble(text);

  // Try each delimiter against both the whole file and the file with its
  // comment block removed, and keep whichever yields the widest header. Width
  // is the right test: every failure here shows up as too few columns.
  let best = null;
  for (const candidate of skipped ? [body, text] : [text]) {
    for (const delimiter of DELIMITERS) {
      const rows = scanDelimited(candidate, delimiter);
      if (!rows.length) continue;
      const width = rows[0].length;
      if (!best || width > best.rows[0].length) best = { rows, delimiter };
    }
  }
  if (!best) return { columns: [], rows: [], delimiter: ",", skipped, problem: null };

  const columns = best.rows[0].map((name) => name.trim());
  const data = best.rows.slice(1)
    .filter((line) => line.length > 1 || (line[0] ?? "").trim() !== "")
    .map((line) => Object.fromEntries(columns.map((name, i) => [name, (line[i] ?? "").trim()])));

  // A one-column table is not an audit. Say so here rather than letting the
  // interface show a single nonsense column and a disabled button.
  const problem = columns.length < 2
    ? `This file parsed as a single column named "${(columns[0] || "").slice(0, 60)}". `
      + `The delimiter is probably not a comma, tab, semicolon or pipe, or the file `
      + `opens with a header block this reader did not recognize. An audit needs a `
      + `target and at least one feature.`
    : null;

  return { columns, rows: data, delimiter: best.delimiter, skipped, problem };
}

/* ══════════ STAGE 1 · OVERVIEW ══════════ */

const dropzone = $("dropzone");
dropzone.addEventListener("click", () => $("file").click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("file").click(); }
});
["dragenter", "dragover"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault(); dropzone.classList.add("over");
  }));
["dragleave", "drop"].forEach((name) =>
  dropzone.addEventListener(name, () => dropzone.classList.remove("over")));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) loadFile(file);
});
$("file").addEventListener("change", (event) => {
  if (event.target.files[0]) loadFile(event.target.files[0]);
});

$("dict-btn").addEventListener("click", () => $("dict-file").click());
$("dict-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  app.dictFile = file;
  $("dict-status").textContent = `attached · ${file.name}`;
  $("dict-btn").textContent = "Change dictionary";
});

$("demo-btn").addEventListener("click", runDemo);
$("brand").addEventListener("click", (event) => { event.preventDefault(); showTab("overview"); });

async function loadFile(file) {
  if (!/\.(csv|parquet)$/i.test(file.name)) {
    $("dz-file").textContent = "that does not look like a CSV or Parquet file";
    return;
  }
  app.file = file;
  app.demo = false;
  $("dz-file").textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;

  if (/\.parquet$/i.test(file.name)) {
    // Parquet is binary; the browser cannot preview it, but the server reads
    // it fine. Say so plainly rather than showing an empty table.
    $("table-note").textContent =
      "Parquet is read by the server, so there is no in-browser preview of the rows.";
    app.table = null;
    unlock("data"); showTab("data");
    return;
  }
  const text = await file.text();
  app.table = parseCsv(text);
  if (!app.table.columns.length) {
    $("dz-file").textContent = "that file has no readable header row";
    return;
  }
  // A parse that produced one column is a failed parse, not a narrow table.
  // Saying so here is the difference between a user fixing their export and a
  // user watching the tool audit a column of joined text.
  if (app.table.problem) {
    $("dz-file").innerHTML = `<span class="dz-problem">${escapeHtml(app.table.problem)}</span>`;
    status("could not read that file", "warn");
    app.table = null;
    return;
  }
  const read = [];
  if (app.table.skipped) {
    read.push(`skipped ${app.table.skipped} line${app.table.skipped === 1 ? "" : "s"} of comment header`);
  }
  if (app.table.delimiter !== ",") {
    read.push(`delimiter ${app.table.delimiter === "\t" ? "tab" : `"${app.table.delimiter}"`}`);
  }
  $("dz-file").textContent = read.length ? `${file.name} — ${read.join(", ")}` : file.name;
  $("topbar-file").textContent =
    `${file.name} · ${app.table.columns.length} col · ${app.table.rows.length.toLocaleString()} rows`;
  status("data loaded", "go");
  renderData();
  unlock("data");
  showTab("data");
}

/* ══════════ STAGE 2 · DATA ══════════ */

const MAX_RENDERED_ROWS = 4000;

function renderData() {
  const { columns, rows } = app.table;
  const shown = Math.min(rows.length, MAX_RENDERED_ROWS);
  $("table-note").textContent = shown < rows.length
    ? `All ${columns.length} columns. Showing the first ${shown.toLocaleString()} of ` +
      `${rows.length.toLocaleString()} rows; every row is still used by the audit.`
    : `All ${columns.length} columns and all ${rows.length.toLocaleString()} rows.`;

  const header = ["<th class=\"rownum\">#</th>"]
    .concat(columns.map((name) => `<th data-col="${escapeHtml(name)}">${escapeHtml(name)}</th>`))
    .join("");
  const body = rows.slice(0, shown).map((row, index) => {
    const cells = columns.map((name) => {
      const value = row[name];
      return value === "" || value === undefined
        ? `<td class="empty" data-col="${escapeHtml(name)}">empty</td>`
        : `<td data-col="${escapeHtml(name)}">${escapeHtml(value)}</td>`;
    }).join("");
    return `<tr><td class="rownum">${index + 1}</td>${cells}</tr>`;
  }).join("");

  $("full-table").innerHTML =
    `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;

  renderCompleteness();
  fillTargetChoices();
}

$("wrap-cells").addEventListener("change", (event) => {
  $("full-table").classList.toggle("wrap", event.target.checked);
});

function completeness() {
  const { columns, rows } = app.table;
  return columns.map((name) => {
    const filled = rows.reduce(
      (total, row) => total + (row[name] !== "" && row[name] !== undefined ? 1 : 0), 0);
    return { name, filled, missing: rows.length - filled, share: rows.length ? filled / rows.length : 0 };
  });
}

let fillSort = "table";
$("fill-sort").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  fillSort = button.dataset.sort;
  $("fill-sort").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === button));
  renderCompleteness();
});

function renderCompleteness() {
  const rowCount = app.table.rows.length;
  let entries = completeness();
  if (fillSort === "empty") entries = [...entries].sort((a, b) => a.share - b.share);

  const head = `<div class="fill-row fill-head">
      <span>column</span><span>filled</span><span style="text-align:right">complete</span>
      <span style="text-align:right">missing</span></div>`;

  $("fill-list").innerHTML = head + entries.map((entry) => {
    const percent = entry.share * 100;
    const tone = percent < 40 ? "low" : percent < 85 ? "mid" : "";
    const isTarget = entry.name === app.target;
    return `<div class="fill-row${isTarget ? " is-target" : ""}">
        <span class="fill-name" title="${escapeHtml(entry.name)}">${escapeHtml(entry.name)}</span>
        <span class="fill-track"><i class="fill-bar ${tone}" style="width:${percent.toFixed(1)}%"></i></span>
        <span class="fill-pct">${percent.toFixed(1)}%</span>
        <span class="fill-missing">${entry.missing.toLocaleString()} of ${rowCount.toLocaleString()}</span>
      </div>`;
  }).join("");
}

/* ── choosing the target ──────────────────────────────────────────────

   The wrong target column is the one mistake that invalidates everything
   downstream, silently: every verdict is "is this knowable before *that*
   column is decided", so a mis-picked target produces a page of confident
   answers to the wrong question. So the column is not a bare name in a list.
   Each option carries what the column actually holds, the likely candidates
   are offered up front, and the choice is echoed back as a class breakdown
   that a reader can recognize or reject at a glance. */

const OUTCOME_NAME = /(survived|target|label|class|outcome|status|result|churn|default|disposition|diagnosis|died|death|readmit|fraud|approved|converted|response|y)$/i;

function columnFacts(name) {
  const rows = app.table.rows;
  const counts = new Map();
  let missing = 0;
  for (const row of rows) {
    const value = row[name];
    if (value === "" || value === undefined || value === null) { missing += 1; continue; }
    counts.set(value, (counts.get(value) || 0) + 1);
  }
  const present = rows.length - missing;
  const values = [...counts.keys()];
  const numeric = values.length > 0 && values.every((value) => Number.isFinite(Number(value)));
  return { name, counts, missing, present, distinct: counts.size, numeric };
}

/* A crude score, deliberately. It orders three chips; it does not decide
   anything, and a wrong guess costs the reader one glance. */
function targetScore(facts) {
  if (!facts.present || facts.distinct < 2) return -1;
  let score = 0;
  if (OUTCOME_NAME.test(facts.name)) score += 6;
  if (facts.distinct <= 10) score += 4;
  else if (facts.distinct <= 25) score += 1;
  else score -= 3;
  if (facts.distinct === 2) score += 2;
  if (facts.missing / (facts.present + facts.missing) > 0.5) score -= 3;
  if (facts.numeric && facts.distinct > 25) score -= 4;
  return score;
}

function fillTargetChoices() {
  const select = $("target");
  const facts = app.table.columns.map(columnFacts);
  const byName = new Map(facts.map((entry) => [entry.name, entry]));
  const ranked = [...facts].sort((a, b) => targetScore(b) - targetScore(a));
  const guess = targetScore(ranked[0]) >= 8 ? ranked[0].name : null;

  select.innerHTML = `<option value="">choose a column…</option>` +
    app.table.columns.map((name) => {
      const entry = byName.get(name);
      const kind = entry.numeric ? "numbers" : "text";
      return `<option value="${escapeHtml(name)}"${name === guess ? " selected" : ""}>`
        + `${escapeHtml(name)} — ${kind}, ${entry.distinct.toLocaleString()} distinct`
        + `${entry.missing ? `, ${entry.missing.toLocaleString()} empty` : ""}</option>`;
    }).join("");

  // Offered, not applied. Anything scoring below the threshold is a column the
  // heuristic has no opinion about, and suggesting it would be noise.
  const likely = ranked.filter((entry) => targetScore(entry) >= 6).slice(0, 3);
  const suggest = $("target-suggest");
  suggest.classList.toggle("hidden", likely.length === 0);
  suggest.innerHTML = likely.length
    ? `<span class="note">likely</span>` + likely.map((entry) =>
        `<button type="button" class="suggest-chip" data-name="${escapeHtml(entry.name)}">
           ${escapeHtml(entry.name)}<span>${entry.distinct} values</span></button>`).join("")
    : "";

  if (guess) app.target = guess;
  markTarget();
  renderTargetPreview();
  validateJob();
}

$("target-suggest").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  $("target").value = button.dataset.name;
  chooseTarget(button.dataset.name);
});

$("target").addEventListener("change", (event) => chooseTarget(event.target.value || null));

function chooseTarget(name) {
  app.target = name;
  markTarget();
  renderCompleteness();
  renderTargetPreview();
  validateJob();
  $("target-suggest").querySelectorAll("button").forEach((button) =>
    button.classList.toggle("on", button.dataset.name === name));
}

/* What the tool will treat as the thing being predicted, said back in the
   reader's own values. A target with four thousand distinct numbers is a
   regression problem and this tool measures classification; better to say so
   here than to let it become a page of results about four thousand classes. */
function renderTargetPreview() {
  const box = $("target-preview");
  if (!app.target || !app.table) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  const facts = columnFacts(app.target);
  box.classList.remove("hidden");

  if (facts.distinct > 20) {
    box.innerHTML = `<div class="tp-warn">
        <b>${facts.distinct.toLocaleString()} distinct values.</b>
        ${facts.numeric
          ? `That reads as a quantity rather than an outcome. Crucible compares
             classification models, so a continuous target needs binning into classes
             before it will mean anything here.`
          : `That reads as an identifier or free text rather than an outcome. Every
             distinct value becomes a class, and most of them will have one row.`}
      </div>`;
    return;
  }

  const ordered = [...facts.counts.entries()].sort((a, b) => b[1] - a[1]);
  const largest = ordered[0]?.[1] || 1;
  const smallest = ordered[ordered.length - 1];
  box.innerHTML = `
    <div class="tp-head">${facts.distinct} classes across
      ${facts.present.toLocaleString()} rows${facts.missing
        ? ` · <b>${facts.missing.toLocaleString()} rows are empty here and cannot be used</b>` : ""}</div>
    ${ordered.map(([value, count]) => `
      <div class="tp-row">
        <span class="tp-name" title="${escapeHtml(value)}">${escapeHtml(value)}</span>
        <span class="tp-track"><i style="width:${((count / largest) * 100).toFixed(1)}%"></i></span>
        <span class="tp-count">${count.toLocaleString()}</span>
        <span class="tp-share">${((count / facts.present) * 100).toFixed(1)}%</span>
      </div>`).join("")}
    ${smallest && smallest[1] < 25 ? `<div class="tp-warn">The smallest class has
      ${smallest[1]} rows. Five-fold cross-validation needs a handful in every fold, so
      anything below about twenty-five makes the comparison noisy.</div>` : ""}`;
}

function markTarget() {
  $("full-table").querySelectorAll("[data-col]").forEach((cell) => {
    cell.classList.toggle("target", cell.dataset.col === app.target);
  });
}

$("prediction-point").addEventListener("input", (event) => {
  app.predictionPoint = event.target.value;
  // The examples are a starting point, so the moment one is edited it stops
  // claiming to be the thing in the box.
  $("examples").querySelectorAll("button").forEach((button) =>
    button.classList.toggle("on", button.dataset.x === event.target.value));
  validateJob();
});

$("examples").querySelectorAll("button").forEach((button) =>
  button.addEventListener("click", () => {
    $("prediction-point").value = button.dataset.x;
    app.predictionPoint = button.dataset.x;
    $("examples").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === button));
    $("prediction-point").focus();
    validateJob();
  }));

/* This sentence is sent to the model verbatim and is the whole basis of every
   timing verdict, so it is worth a moment's feedback rather than a silent
   accept. Nothing here blocks: a reader who wants to write two words may, and
   the note says what that costs. */
const POINT_MINIMUM = 8;

function pointQuality(text) {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  if (words < 3) {
    return { tone: "thin", text: "A few more words will do more work: name the event, and "
      + "what has not happened yet at that moment." };
  }
  if (!/\b(before|prior to|not yet|without|at the (time|moment|point)|when|as of|upon|during|at)\b/i.test(text)) {
    return { tone: "thin", text: "Anchor it to a moment — “at”, “when”, “before” — so a column "
      + "can be judged against something." };
  }
  if (!/\b(before|prior to|not yet|without|no .* yet)\b/i.test(text)) {
    return { tone: "ok", text: "Good. Saying what has <em>not</em> happened yet at that moment "
      + "sharpens it further: that clause is what catches consequence columns." };
  }
  return { tone: "good", text: "This says when the prediction happens and what is not yet known. "
    + "That is exactly what the screen needs." };
}

function validateJob() {
  const point = app.predictionPoint.trim();
  const ready = Boolean(app.target) && point.length > POINT_MINIMUM;
  $("to-model").disabled = !ready;

  const check = $("point-check");
  if (point.length > POINT_MINIMUM) {
    const quality = pointQuality(point);
    check.className = `point-check ${quality.tone}`;
    check.innerHTML = quality.text;
  } else {
    check.className = "point-check hidden";
    check.innerHTML = "";
  }

  $("job-note").textContent = !app.target
    ? "Choose the column you are predicting."
    : point.length <= POINT_MINIMUM
      ? "Describe when the prediction happens — a phrase is enough."
      : "";
}

$("to-model").addEventListener("click", () => { unlock("model"); showTab("model"); loadModels(); });

/* ══════════ STAGE 3 · MODEL ══════════ */

let modelsLoaded = false;

async function loadModels() {
  if (modelsLoaded) return;
  modelsLoaded = true;
  try {
    const response = await fetch("api/models");
    const data = await response.json();
    app.model = app.model || data.default;
    app.modelCatalogue = data.models;
    $("model-list").innerHTML = data.models.map((model) => {
      const measured = model.measured;
      const locked = Boolean(model.needs_key) && model.needs_key !== app.keyProvider;
      return `
      <label class="model-card${model.id === data.default ? " on" : ""}${locked ? " locked" : ""}">
        <input type="radio" name="model" value="${escapeHtml(model.id)}"
          ${model.id === data.default ? "checked" : ""}${locked ? " disabled" : ""}>
        <b>${escapeHtml(model.name)}</b>
        ${model.recommended ? '<span class="model-tag">default</span>'
          : locked ? `<span class="model-tag locked">needs ${providerLabel(model.needs_key)} key in Settings</span>`
          : "<span></span>"}
        <p><span class="model-size">${escapeHtml(model.size)}</span> — ${escapeHtml(model.note)}</p>
        ${measured ? `<p class="model-measured">Measured on the benchmark:
          <b>F1 ${measured.f1}</b> · precision ${measured.precision} · recall ${measured.recall}
          <span>over ${measured.datasets} datasets at ${measured.shuffles}
          shuffle${measured.shuffles === 1 ? "" : "s"}, under the same wording of the
          criterion this tool sends it</span></p>` : ""}
      </label>`;
    }).join("");
    $("model-list").addEventListener("change", (event) => {
      app.model = event.target.value;
      $("model-list").querySelectorAll(".model-card").forEach((card) =>
        card.classList.toggle("on", card.querySelector("input").checked));
    });
    app.serverKey = data.server_key_present;
    app.demoPool = data.demo_pool;

    // The shared pool exists so the tool can be tried without an account. Say
    // exactly how much of it is left, because a visitor who finds out by being
    // refused mid-audit has had a worse time than one who was told first.
    const pool = data.demo_pool;
    const status = $("pool-status");
    if (!pool) {
      status.className = "pool-status spent";
      status.textContent = "No shared pool is configured on this server, so an audit needs your own key.";
    } else if (pool.remaining_today > 0) {
      status.className = "pool-status";
      status.innerHTML =
        `<b>${pool.remaining_today}</b> of ${pool.capacity_today} shared audits left today, ` +
        `across ${pool.keys} pooled key${pool.keys === 1 ? "" : "s"}. This is what runs if ` +
        `you leave the key below empty. Quota is spent least-used-first, so one visitor ` +
        `cannot drain it, and it refills at midnight UTC.`;
    } else {
      status.className = "pool-status spent";
      status.innerHTML =
        `The shared pool is spent for today. Add your own key below to continue; it ` +
        `refills at midnight UTC.`;
    }

    const usable = Boolean(app.apiKey) || (pool && pool.remaining_today > 0);
    $("run-note").textContent = usable
      ? "" : "The shared pool is spent. Add your own key in Settings to continue.";
    $("run-btn").disabled = !usable;
  } catch (error) {
    // Allow a later attempt: a transient failure should not leave the page
    // permanently convinced it has already loaded.
    modelsLoaded = false;
    const message = `Could not reach the server: ${escapeHtml(error.message)}`;
    $("model-list").innerHTML = `<p class="note">${message}</p>`;
    const status = $("pool-status");
    if (status) {
      status.className = "pool-status spent";
      status.textContent = "Could not reach the server to check the shared pool.";
    }
  }
}

/* Vendor names as a reader would write them, with the right article. The API
   returns a slug, and "needs a openai key" is the kind of detail that makes a
   tool look unfinished. */
const PROVIDER_LABEL = {
  anthropic: "an Anthropic", openai: "an OpenAI",
  featherless: "a Featherless", gemini: "a Gemini",
};
function providerLabel(slug) {
  return PROVIDER_LABEL[slug] || `a ${slug}`;
}

/* One field for every provider. Which vendor issued a key is decided from its
   own prefix, server-side, so nothing is sent to any provider to find out and
   no request is spent. Debounced, because this fires on every keystroke. */
let keyCheckTimer = null;

$("api-key").addEventListener("input", (event) => {
  app.apiKey = event.target.value.trim();
  const pooled = app.demoPool && app.demoPool.remaining_today > 0;
  $("run-btn").disabled = !(app.apiKey || pooled);
  $("run-note").textContent = "";

  clearTimeout(keyCheckTimer);
  if (!app.apiKey) {
    app.keyProvider = null;
    $("key-detect").textContent = "";
    $("key-note").textContent = "Empty, so audits draw on the shared Gemini pool.";
    modelsLoaded = false;
    loadModels();
    return;
  }
  $("key-detect").textContent = "Reading the key\u2019s prefix\u2026";
  keyCheckTimer = setTimeout(() => identifyKey(app.apiKey), 350);
});

async function identifyKey(key) {
  try {
    const form = new FormData();
    form.append("api_key", key);
    const response = await fetch("api/key/check", { method: "POST", body: form });
    const result = await response.json();
    if (key !== app.apiKey) return;          // superseded by later typing
    const detect = $("key-detect");

    if (result.recognized) {
      app.keyProvider = result.provider;
      detect.className = "note key-ok";
      detect.textContent = result.message;
      $("key-note").textContent =
        "Used for every audit this session. The shared pool stays untouched.";
    } else {
      // A prefix table describes what keys looked like when it was written.
      // Google has already changed theirs once. An unrecognized key is a key
      // whose owner gets asked, never one that is refused.
      app.keyProvider = null;
      detect.className = "note key-unknown";
      detect.innerHTML = `${escapeHtml(result.message)}
        <span class="key-pick">${(result.providers || []).map((name) => `
          <button type="button" class="key-pick-btn" data-provider="${escapeHtml(name)}">
            ${escapeHtml(providerLabel(name).replace(/^an? /, ""))}
          </button>`).join("")}</span>`;
      detect.querySelectorAll(".key-pick-btn").forEach((button) => {
        button.addEventListener("click", () => {
          app.keyProvider = button.dataset.provider;
          detect.className = "note key-ok";
          detect.textContent =
            `Treating this as ${providerLabel(app.keyProvider)} key. `
            + `If the provider rejects it, the audit will say so.`;
          $("key-note").textContent =
            "Used for every audit this session. The shared pool stays untouched.";
          modelsLoaded = false;
          loadModels();
        });
      });
      $("key-note").textContent = "";
    }
    modelsLoaded = false;
    loadModels();          // a usable key unlocks models, so redraw the list
  } catch (error) {
    $("key-detect").textContent = "Could not reach the server to check the key.";
  }
}

$("run-btn").addEventListener("click", startAudit);

/* ══════════ STAGE 4 · DETECTION ══════════ */

const STAGES = [
  ["semantic", "Reading every column name against the target"],
  ["statistical", "Measuring correlation with the target"],
  ["triage", "Sorting columns by which screens flagged them"],
  ["contested", "Checking flagged columns against the documentation"],
];

/* An audit is a model writing a verdict for every column, so how long it takes
   is a function of how wide the table is. Saying so beats a spinner: a person
   who knows it will be two minutes goes and does something else, and a person
   who does not sits watching a bar and decides the tool has hung. */
function expectedDuration(columnCount) {
  if (columnCount <= 20) return "usually under a minute";
  if (columnCount <= 60) return "usually one to two minutes";
  if (columnCount <= 120) return "usually two to four minutes";
  return "several minutes on a table this wide";
}

let elapsedTimer = null;

function startElapsed() {
  const began = Date.now();
  const columns = app.table?.columns?.length || 0;
  const estimate = expectedDuration(columns);
  const note = $("run-eta");
  if (!note) return;

  const tick = () => {
    const seconds = Math.round((Date.now() - began) / 1000);
    const clock = seconds < 60
      ? `${seconds}s`
      : `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
    note.textContent = `${clock} elapsed \u00b7 ${estimate} for ${columns} columns. `
      + `Nothing is lost if you switch tabs.`;
  };
  tick();
  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(tick, 1000);
}

function stopElapsed(finalNote) {
  clearInterval(elapsedTimer);
  elapsedTimer = null;
  const note = $("run-eta");
  if (note) note.textContent = finalNote || "";
}

function paintStages(activeIndex) {
  $("run-stages").innerHTML = STAGES.map(([key, label], index) => {
    const stateName = index < activeIndex ? "done" : index === activeIndex ? "now" : "";
    return `<div class="run-stage ${stateName}"><i>${index < activeIndex ? "✓" : index + 1}</i>${label}</div>`;
  }).join("");
}

const setProgress = (fraction) =>
  $("progress-fill").style.width = `${Math.min(100, fraction * 100)}%`;

async function startAudit() {
  unlock("detect");
  showTab("detect");
  status("running", "busy");
  $("error-banner").classList.add("hidden");
  $("found-panel").classList.add("hidden");
  $("run-state").textContent = "Uploading and starting the audit…";
  paintStages(0);
  setProgress(0.04);
  startElapsed();

  const form = new FormData();
  form.append("file", app.file);
  form.append("target", app.target);
  form.append("prediction_point", app.predictionPoint);
  form.append("model", app.model);
  // One key. The server checks it was issued by the vendor serving the chosen
  // model and refuses otherwise, rather than forwarding column names to a
  // company the user did not pick.
  if (app.apiKey) form.append("api_key", app.apiKey);
  if (app.dictFile) form.append("dictionary", app.dictFile);

  let started;
  try {
    const response = await fetch("api/audit", { method: "POST", body: form });
    started = await response.json();
    if (!response.ok) throw new Error(started.detail || "the audit could not be started");
  } catch (error) {
    return failAudit(error.message);
  }
  app.jobId = started.job_id;
  if (started.dictionary) {
    $("dict-status").textContent =
      `${started.dictionary.matched} of ${app.table.columns.length - 1} columns documented`;
  }
  followAudit();
}

function failAudit(message) {
  status("failed", "bad");
  setProgress(0);
  stopElapsed();
  $("run-state").textContent = "The audit stopped.";
  $("error-banner").textContent = message;
  $("error-banner").classList.remove("hidden");
}

function followAudit() {
  const stream = new EventSource(`api/audit/${app.jobId}/events`);
  let stageIndex = 0;

  stream.addEventListener("stage", (message) => {
    const data = JSON.parse(message.data);
    stageIndex = Math.max(stageIndex, STAGES.findIndex(([key]) => key === data.stage));
    paintStages(stageIndex);
    setProgress(0.1 + stageIndex * 0.22);
    $("run-state").textContent = STAGES[stageIndex]?.[1] || "Working…";
  });

  stream.addEventListener("shuffle", (message) => {
    const data = JSON.parse(message.data);
    $("run-state").textContent =
      `Reading every column name against the target — pass ${data.shuffle + 1} of ${data.of || 3}`;
    setProgress(0.1 + ((data.shuffle + 1) / (data.of || 3)) * 0.2);
  });

  stream.addEventListener("done", async () => {
    stream.close();
    paintStages(STAGES.length);
    setProgress(1);
    stopElapsed();
    await loadAudit();
  });

  stream.addEventListener("error", (message) => {
    stream.close();
    let detail = "the connection to the audit was lost";
    try { detail = JSON.parse(message.data).message; } catch (_) { /* stream closed */ }
    failAudit(detail);
  });
}

async function loadAudit() {
  const response = await fetch(`api/audit/${app.jobId}`);
  app.audit = await response.json();
  if (app.audit.error) return failAudit(app.audit.error);
  status("audit complete", "go");
  $("run-state").textContent = "Both screens finished.";
  markAudited();
  // The checklist and the progress bar are still settling when the result
  // arrives, and results landing on top of a half-finished animation reads as
  // a glitch rather than as an answer. Let the run visibly finish first.
  await settled();
  showFindings();
}

/* Wait for the progress animation to land, then a beat. Ties to the CSS
   transition rather than a guessed number, and honours a reader who has asked
   the system for less motion by not waiting at all. */
function settled(beat = 260) {
  const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (reduced) return Promise.resolve();
  const fill = $("progress-fill");
  const bar = new Promise((resolve) => {
    if (!fill) return resolve();
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    fill.addEventListener("transitionend", finish, { once: true });
    setTimeout(finish, 700);              // never hang on a transition that
  });                                      // never fires
  return bar.then(() => new Promise((resolve) => setTimeout(resolve, beat)));
}

function showFindings() {
  const buckets = app.audit.buckets || {};
  const counts = { A: 0, B: 0, C: 0, D: 0 };
  Object.values(buckets).forEach((bucket) => counts[bucket]++);
  const total = Object.keys(buckets).length;

  $("found-note").textContent =
    `${total} columns judged against ${app.audit.target}.` +
    (app.audit.dictionary ? " Verdicts are grounded in your data dictionary." : "");

  const cards = [
    ["Both screens", counts.A,
     "Flagged by the model and by correlation. Near certain.", "flagged"],
    ["Model only", counts.B,
     "A leak correlation cannot see. This is the group statistics would have missed, and the reason this tool exists.", "payoff"],
    ["Statistics only", counts.C,
     "Correlated strongly but judged legitimate. Usually a real predictor.", "quiet"],
    ["Neither", counts.D, "Nothing flagged these.", "quiet"],
  ];
  $("found-grid").innerHTML = cards.map(([label, value, note, tone]) => `
    <div class="found-card ${tone}${tone === "payoff" && value ? " lit" : ""}">
      <span>${label}</span><strong>${value}</strong><small>${note}</small>
    </div>`).join("");

  $("found-panel").classList.remove("hidden");
  unlock("review");
  renderReview();
}

$("to-review").addEventListener("click", () => showTab("review"));

/* ══════════ STAGE 5 · REVIEW ══════════ */

let reviewFilter = "all";

function reviewRows() {
  const columns = app.audit.schema.feature_columns;
  return columns.map((column) => {
    const semantic = app.audit.semantic?.[column] || {};
    const statistical = app.audit.statistical?.[column] || {};
    const bucket = app.audit.buckets?.[column] || "D";
    return {
      column,
      verdict: semantic.verdict || "OK",
      mechanism: semantic.mechanism,
      reasons: semantic.reasons || [],
      votes: semantic.leak_votes || 0,
      passes: semantic.shuffles_counted || 3,
      correlation: statistical.correlation,
      missingnessGap: statistical.missingness_gap,
      univariateAuc: statistical.auc,
      flaggedByStats: Boolean(statistical.flagged),
      bucket,
      description: (app.audit.descriptions || {})[column],
      contested: Boolean(semantic.contested),
      contestedReason: semantic.contested_reason,
    };
  }).sort((a, b) => {
    const rank = (row) => (row.verdict === "LEAK" ? 0 : row.verdict === "ABSTAIN" ? 1 : 2);
    return rank(a) - rank(b) || a.column.localeCompare(b.column);
  });
}

const BUCKET_LABEL = {
  A: ["both screens", ""], B: ["model only", "model-only"],
  C: ["statistics only", ""], D: ["neither", ""],
};

function renderReview() {
  const rows = reviewRows();
  const counts = {
    all: rows.length,
    flagged: rows.filter((row) => row.verdict === "LEAK").length,
    "model-only": rows.filter((row) => row.bucket === "B").length,
    clean: rows.filter((row) => row.verdict === "OK").length,
  };
  $("filters").innerHTML = [
    ["all", "all"], ["flagged", "flagged"], ["model-only", "model only"], ["clean", "cleared"],
  ].map(([key, label]) =>
    `<button type="button" data-f="${key}" class="${key === reviewFilter ? "on" : ""}">${label}<b>${counts[key]}</b></button>`
  ).join("");

  const visible = rows.filter((row) =>
    reviewFilter === "all" ? true
      : reviewFilter === "flagged" ? row.verdict === "LEAK"
      : reviewFilter === "model-only" ? row.bucket === "B"
      : row.verdict === "OK");

  $("evidence").innerHTML =
    `<div class="ev-head"><span>column</span><span>verdict</span><span>orders</span>
       <span>|r|</span><span>agreement</span><span>decision</span></div>` +
    visible.map(evidenceRow).join("");

  $("review-count").textContent =
    `${counts.flagged} of ${counts.all} columns flagged as leaks, ${counts["model-only"]} of them invisible to correlation.`;
  updateReviewFoot();
}

function evidenceRow(row) {
  const isLeak = row.verdict === "LEAK";
  const pill = row.contested ? "contested"
    : isLeak ? "leak" : row.verdict === "ABSTAIN" ? "abstain" : "clean";
  const label = row.contested ? "CONTESTED"
    : isLeak ? (row.mechanism || "LEAK") : row.verdict === "ABSTAIN" ? "ABSTAIN" : "CLEAN";
  const dots = Array.from({ length: row.passes }, (_, index) =>
    `<i class="${index < row.votes ? "on" : ""}"></i>`).join("");
  const magnitude = row.correlation === null || row.correlation === undefined
    ? null : Math.abs(row.correlation);
  const [agreeText, agreeClass] = BUCKET_LABEL[row.bucket];
  const decision = app.decisions[row.column];

  // Every column carries the model's reasoning, cleared ones included. A
  // verdict of "fine" is still a judgement, and the reader deserves to see
  // what it rested on rather than only being told when something is wrong.
  const headline = row.reasons[0]
    || (isLeak ? "flagged as a leak, no specific wording returned"
               : "judged available at the prediction point");

  return `<div class="ev-row ${isLeak ? "leak" : ""}" data-column="${escapeHtml(row.column)}">
      <span class="ev-name">${escapeHtml(row.column)}
        <em class="ev-reason">${escapeHtml(headline)}</em></span>
      <span><span class="pill ${pill}">${escapeHtml(label)}</span></span>
      <span class="dots">${dots}</span>
      <span class="corr">
        <span class="corr-value">${magnitude === null ? "—" : magnitude.toFixed(2)}</span>
        <span class="corr-track"><i class="corr-fill ${magnitude > 0.5 ? "over" : ""}"
          style="width:${magnitude === null ? 0 : Math.min(100, magnitude * 100)}%"></i></span>
      </span>
      <span class="agree ${agreeClass}">${agreeText}</span>
      <span class="decide">
        <button type="button" data-d="drop" class="${decision === "drop" ? "on" : ""}">drop</button>
        <button type="button" data-d="keep" class="${decision === "keep" ? "on" : ""}">keep</button>
      </span>
    </div>`;
}

$("filters").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  reviewFilter = button.dataset.f;
  renderReview();
});

$("evidence").addEventListener("click", (event) => {
  const decideButton = event.target.closest(".decide button");
  const row = event.target.closest(".ev-row");
  if (!row) return;

  if (decideButton) {
    event.stopPropagation();
    const column = row.dataset.column;
    app.decisions[column] =
      app.decisions[column] === decideButton.dataset.d ? undefined : decideButton.dataset.d;
    row.querySelectorAll(".decide button").forEach((button) =>
      button.classList.toggle("on", button.dataset.d === app.decisions[column]));
    updateReviewFoot();
    return;
  }
  toggleExpand(row);
});

function toggleExpand(row) {
  const existing = row.nextElementSibling;
  if (existing?.classList.contains("expand")) { existing.remove(); return; }
  const data = reviewRows().find((entry) => entry.column === row.dataset.column);
  if (!data) return;

  const mechanism = data.mechanism && MECHANISM_TEXT[data.mechanism]
    ? `<li><b>${data.mechanism}</b> — this column ${MECHANISM_TEXT[data.mechanism]}.</li>` : "";
  const reasons = data.reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("");
  const passes = `<p class="stat-note">Read ${data.passes} times in ${data.passes} different
    column orders; ${data.votes} of those passes called it a leak.</p>`;
  const quote = data.description
    ? `<p class="quote"><span>from your data dictionary</span>${escapeHtml(data.description)}</p>` : "";
  const contestedNote = data.contested
    ? `<p class="quote contested-note"><span>contested &mdash; yours to decide</span>` +
      `The documentation indicates this value is already fixed at the prediction point` +
      `${data.contestedReason ? `: ${escapeHtml(data.contestedReason)}` : "."} ` +
      `So it could honestly have been obtained. Whether it <em>should</em> be used is a ` +
      `different question, and not one this tool will answer for you.</p>` : "";

  const magnitude = data.correlation === null || data.correlation === undefined
    ? null : Math.abs(data.correlation);
  let statNote = "";
  if (data.verdict === "LEAK" && magnitude !== null && magnitude <= 0.5) {
    statNote = `Correlation is ${magnitude.toFixed(3)}, well under the 0.5 threshold, so no ` +
      `statistical screen would have reached this column.`;
  } else if (data.verdict === "LEAK" && magnitude === null) {
    statNote = "Correlation is undefined for this column, so a statistical screen has nothing to test.";
  } else if (data.verdict === "OK" && data.flaggedByStats) {
    statNote = `Correlation is ${magnitude.toFixed(3)}, above the threshold, but the column is ` +
      `judged a legitimate predictor rather than a cause of the label.`;
  }

  const others = [];
  if (data.missingnessGap !== null && data.missingnessGap !== undefined) {
    others.push(`is missing at very different rates between the classes ` +
      `(gap ${data.missingnessGap.toFixed(3)})`);
  }
  if (data.univariateAuc !== null && data.univariateAuc !== undefined) {
    others.push(`ranks the outcome on its own at AUC ${data.univariateAuc.toFixed(3)}`);
  }
  const statLine = others.length
    ? `<p class="stat-note">Other statistics: this column ${others.join(", and ")}.</p>` : "";

  const panel = document.createElement("div");
  panel.className = "expand";
  panel.innerHTML =
    `<ul>${mechanism}${reasons || "<li>No specific reasoning was returned for this column.</li>"}</ul>` +
    quote + contestedNote + (statNote ? `<p class="stat-note">${statNote}</p>` : "") + statLine + passes;
  row.after(panel);
}

$("accept-all").addEventListener("click", () => {
  reviewRows().forEach((row) => {
    if (row.verdict === "LEAK") app.decisions[row.column] = "drop";
  });
  renderReview();
});

function updateReviewFoot() {
  const drops = Object.entries(app.decisions).filter(([, d]) => d === "drop").map(([c]) => c);
  $("impact-btn").disabled = drops.length === 0;
  $("skip-impact").disabled = drops.length === 0;
  if (!drops.length) {
    $("review-note").textContent =
      "Confirm at least one column before running the comparison.";
    return;
  }
  // The cost, before it is paid. The comparison fits ninety models and on a
  // wide table that is minutes, which is worth knowing before clicking rather
  // than after: the drops are already decided, and this stage only measures
  // what they were worth.
  const seconds = estimateFitSeconds();
  $("review-note").textContent =
    `${drops.length} column${drops.length === 1 ? "" : "s"} confirmed for removal: `
    + `${drops.join(", ")}.`
    + (seconds ? ` The comparison fits 90 models and takes ${humanDuration(seconds)} `
                 + `on a table this size. It is optional.` : "");
}

$("impact-btn").addEventListener("click", () => runImpact());
$("skip-impact").addEventListener("click", () => {
  unlock("results");
  showTab("results");
  showSkipped("not run");
});

/* The comparison is the expensive stage and the only optional one: the drops
   are decided by the time a reader gets here, and this measures what they were
   worth rather than deciding anything. Somebody who only wants the cleaned file
   should not have to sit through ninety fits to reach the download. */
function showSkipped(reason) {
  app.impact = null;
  measuredSet = null;
  status("ready", "");
  const drops = currentDrops();
  $("results").innerHTML = `
    <div class="panel skipped">
      <span class="eyebrow">Stage 6 · ${escapeHtml(reason)}</span>
      <h3>The downstream comparison was not run.</h3>
      <p class="sub">Your ${drops.length} confirmed drop${drops.length === 1 ? "" : "s"}
        ${drops.length === 1 ? "is" : "are"} applied to the file below, and the audit report
        carries every verdict and its reason. What is missing is the measurement of what those
        columns were worth: the fits that put a number on the inflation.</p>
      <div class="skipped-actions">
        <button type="button" class="primary" id="run-anyway">Run the comparison after all</button>
        <span class="note">${escapeHtml(humanDuration(estimateFitSeconds() || 0))}, and nothing
          below changes while it runs.</span>
      </div>
    </div>`;
  $("run-anyway").addEventListener("click", () => runImpact());
  renderEditor();
  renderResultTable();
}

// Set while a comparison is in flight, so the stop control has something to
// pull on. Aborting drops the response; the server finishes the work it began
// and throws the answer away.
let fitAbort = null;

async function runImpact({ keepEditor = false } = {}) {
  unlock("results");
  if (!keepEditor) showTab("results");
  status("measuring", "busy");

  // Watch the fit rather than spin at it. The stream has to be open before the
  // request that produces the events, so the run identifier is minted here and
  // travels with the request.
  const runId = `fit-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const fit = startFitView(runId);
  fitAbort = new AbortController();
  $("fit-skip")?.addEventListener("click", () => {
    $("fit-skip").disabled = true;
    $("fit-skip").textContent = "stopping…";
    fitAbort.abort();
  }, { once: true });
  // The server can only publish to a stream that exists, so the request that
  // produces the events must not overtake the stream that carries them. Bounded,
  // because a measurement that runs without its visualization is still a
  // measurement and is not worth blocking.
  await fit.ready;

  const drops = currentDrops();
  let result;
  try {
    // Starting the measurement and collecting it are two requests on purpose.
    // Fitting ninety models takes minutes, and a hosted deployment sits behind
    // a proxy that gives up on a request long before that and answers with its
    // own 502 page. So the post returns at once and the answer is polled for.
    let poll;
    if (app.demo) {
      const started = await readResult(await fetch("api/demo/impact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ drop_list: drops, run_id: runId }),
        signal: fitAbort.signal,
      }));
      poll = `api/demo/impact/${encodeURIComponent(started.run_id || runId)}`;
    } else {
      const verdicts = Object.fromEntries(
        Object.entries(app.decisions).filter(([, decision]) => decision));
      await fetch(`api/audit/${app.jobId}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(verdicts),
      });
      await readResult(await fetch(
        `api/audit/${app.jobId}/impact?run_id=${encodeURIComponent(runId)}`,
        { method: "POST", signal: fitAbort.signal }));
      poll = `api/audit/${app.jobId}/impact`;
    }
    result = await collectImpact(poll, fitAbort.signal);
  } catch (error) {
    fit.close();
    // Stopping is a choice, not a failure, and must not be reported as one.
    if (error.name === "AbortError") return showSkipped("stopped");
    return failImpact(error.message, error);
  }
  fit.close();
  // Rendering can throw too, and when it does the failure looks identical to a
  // failed request from the outside. Catching it here is what tells the two
  // apart instead of leaving a half-drawn page and an unhandled rejection.
  try {
    finishImpact(result);
  } catch (error) {
    failImpact(`the results could not be drawn: ${error.message}`, error);
  }
}

/* ══════════ THE FIT, WHILE IT HAPPENS ══════════

   Every number on this screen comes from models fitted on the spot, and this is
   those models fitting. Three things are drawn, all of them read off the real
   run rather than staged:

   the arm and fold currently being fitted;
   one real tree, node by node, taken off the fitted estimator;
   and what the first forty trees of each arm split on *first*.

   The third is the one worth watching. A forest still holding a leaked column
   reaches for it at the root over and over; the same forest with that column
   removed spreads across ordinary features. That is this tool's entire claim,
   showing up in the structure of the model instead of in a score. */

const ARM_TITLE = {
  with_leaks: "Every column", honest: "Your confirmed drops removed",
  baseline: "What correlation would have removed",
};
const ARM_KIND = { with_leaks: "leaked", honest: "honest", baseline: "baseline" };

/* ── how long this will take ──────────────────────────────────────────

   Measured rather than felt. Two runs of the full protocol, same machine,
   same code:

     1,309 rows × 16 encoded features · 60 fits · 26.8s → 0.45s a fit
     5,000 rows × 51 encoded features · 60 fits · 219s  → 3.65s a fit

   which is a fixed cost per fit plus a term in rows × features. It is a
   straight line through two points and it will be wrong on somebody else's
   hardware, so it is only ever the opening number: once folds start landing,
   `projectRemaining` throws it away and uses the rate this machine is
   actually managing. */
const FIT_FIXED_SECONDS = 0.15;
const FIT_SECONDS_PER_CELL = 1.6e-5;
const secondsPerFit = (rows, features) =>
  FIT_FIXED_SECONDS + rows * features * FIT_SECONDS_PER_CELL;

/* Mirrors `_encode`, which is the only reason this is worth doing in the
   browser at all: numbers pass through as one column, text with twenty or
   fewer distinct values becomes one column per value, and wider text becomes a
   single integer code. Guessing "one column per distinct value" for all text
   overstates a table like Titanic by eight times, because `name` and `ticket`
   are codes rather than categories. */
function estimateEncodedWidth() {
  if (!app.table) return 0;
  const rows = app.table.rows;
  return app.table.columns
    .filter((name) => name !== app.target)
    .reduce((total, name) => {
      const values = [];
      for (const row of rows) {
        const value = row[name];
        if (value !== "" && value !== undefined && value !== null) values.push(value);
      }
      if (!values.length) return total + 1;
      if (values.every((value) => Number.isFinite(Number(value)))) return total + 1;
      const distinct = new Set(values).size;
      return total + (distinct <= 20 ? distinct : 1);
    }, 0);
}

const MAX_FIT_ROWS = 5000;      // impact.MAX_ROWS; wider tables are sampled down

/* What the comparison will cost, before it is started. `arms` is 2 without a
   correlation baseline and 3 with one; `learners` and `configs` and `folds`
   are fixed by the protocol. */
function estimateFitSeconds({ arms = 3 } = {}) {
  const rows = Math.min(app.table?.rows?.length || 0, MAX_FIT_ROWS);
  const features = estimateEncodedWidth();
  if (!rows || !features) return null;
  const fits = arms * 5 * 6;    // arms × folds × (3 configs × 2 learner families)
  return fits * secondsPerFit(rows, features);
}

/* Times are read, not calculated, so they are rounded to something a person
   would say out loud. */
function humanDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "a moment";
  if (seconds < 45) return `${Math.max(5, Math.round(seconds / 5) * 5)} seconds`;
  if (seconds < 90) return "about a minute";
  const minutes = seconds / 60;
  if (minutes < 10) {
    const low = Math.floor(minutes);
    return `about ${low} to ${low + 1} minutes`;
  }
  return `about ${Math.round(minutes)} minutes`;
}

const clock = (seconds) => seconds < 60
  ? `${Math.round(seconds)}s`
  : `${Math.floor(seconds / 60)}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;

function startFitView(runId) {
  const state = { plan: null, arms: new Map(), current: null, done: 0, total: 0,
                  trees: [] };
  app.fitTrees = state.trees;      // kept for the gallery on the results tab
  $("results").innerHTML = `
    <div class="panel fitview" id="fitview">
      <div class="fit-head">
        <div>
          <span class="eyebrow">Stage 6 · fitting</span>
          <h3 id="fit-title">Starting the comparison…</h3>
          <p class="sub" id="fit-sub">Two learner families, a hyperparameter search on every
            arm, five folds each. Same folds, same encoding, same search, so no arm is a
            strawman.</p>
        </div>
        <div class="fit-progress">
          <div class="fit-bar"><i id="fit-bar-fill"></i></div>
          <span class="fit-count" id="fit-count"></span>
          <span class="fit-eta" id="fit-eta">estimating…</span>
          <button type="button" class="ghost small" id="fit-skip">Stop and just download the CSV</button>
        </div>
      </div>
      <div class="fit-body">
        <div class="fit-tree">
          <h4>A tree this model just built</h4>
          <p class="sub" id="tree-caption">Real structure, read off the fitted estimator.</p>
          <svg id="fit-tree-svg" viewBox="0 0 420 250" role="img"
               aria-label="a decision tree from the model being fitted"></svg>
        </div>
        <div class="fit-census">
          <h4>What each arm reaches for first</h4>
          <p class="sub">The root split of the first forty trees. A leaked column shows up
            here before any score is computed.</p>
          <div id="fit-census-body"></div>
        </div>
      </div>
    </div>`;

  let source = null;
  try {
    source = new EventSource(`api/fit/${encodeURIComponent(runId)}/events`);
  } catch (error) {
    return { ready: Promise.resolve(), close() {} };
  }
  const ready = new Promise((resolve) => {
    source.addEventListener("open", resolve, { once: true });
    source.addEventListener("error", resolve, { once: true });
    setTimeout(resolve, 2500);
  });

  /* The remaining time, from the rate this run is actually managing.

     The opening figure is the estimate from two calibration runs, which is a
     guess about somebody else's hardware. From the third fold onwards it is
     replaced by measurement: elapsed divided by fits completed, times fits
     left. The two are blended over the first ten so the number does not lurch
     when the first fold happens to be quick. */
  const began = Date.now();
  const projectRemaining = () => {
    if (!state.total || !state.done) return state.priorSeconds || null;
    const elapsed = (Date.now() - began) / 1000;
    const measured = (elapsed / state.done) * (state.total - state.done);
    if (!state.priorSeconds) return measured;
    const trust = Math.min(1, state.done / 10);
    const priorLeft = state.priorSeconds * (1 - state.done / state.total);
    return measured * trust + priorLeft * (1 - trust);
  };

  const bump = () => {
    if (!state.total) return;
    const share = Math.min(1, state.done / state.total);
    $("fit-bar-fill").style.width = `${(share * 100).toFixed(1)}%`;
    $("fit-count").textContent = `${state.done} of ${state.total} fits`;
    const left = projectRemaining();
    const eta = $("fit-eta");
    if (eta) {
      eta.textContent = left === null ? "estimating…"
        : state.done >= state.total ? "finishing up"
        : `about ${clock(left)} left`;
    }
  };
  const ticker = setInterval(bump, 1000);

  source.addEventListener("plan", (event) => {
    state.plan = JSON.parse(event.data);
    const configs = Object.values(state.plan.configs).reduce((a, b) => a + b, 0);
    // Arms sharing a column set are fitted once, so counting all of them would
    // leave the bar permanently short of its own finish.
    const fitted = state.plan.distinct_arms || state.plan.arms.length;
    state.total = fitted * state.plan.folds * configs;
    const width = Math.max(...Object.values(state.plan.features || {}), 1);
    state.priorSeconds = state.total * secondsPerFit(state.plan.rows, width);
    $("fit-sub").textContent =
      `${state.plan.rows.toLocaleString()} rows · ${state.plan.arms.length} arms · ` +
      `${state.plan.learners.length} learner families · ${state.plan.folds} folds each. ` +
      `Identical folds and an identical search on every arm.`;
    renderCensus(state);
    bump();
  });

  source.addEventListener("arm_start", (event) => {
    const data = JSON.parse(event.data);
    state.current = data;
    $("fit-title").textContent =
      `${ARM_TITLE[data.arm] || data.arm} · ${data.learner.replace(/_/g, " ")}`;
  });

  source.addEventListener("arm_reused", (event) => {
    const data = JSON.parse(event.data);
    $("fit-title").textContent =
      `${ARM_TITLE[data.arm] || data.arm} — identical to an arm already fitted, so not refitted`;
  });

  source.addEventListener("fold", () => { state.done += 1; bump(); });

  source.addEventListener("tree", (event) => {
    const data = JSON.parse(event.data);
    if (data.root_census?.length) state.arms.set(data.arm, data);
    renderCensus(state);
    // Every tree is kept. During the fit the newest is shown, and afterwards
    // the gallery lets a reader step through all of them: an arm that still
    // holds a leaked column and the same arm without it are the comparison
    // this whole tool is about, and one frame of it is not enough to see.
    state.trees.push(data);
    drawTree(data, $("fit-tree-svg"), $("tree-caption"));
  });

  source.addEventListener("done", () => { source.close(); });
  source.addEventListener("failed", () => { source.close(); });

  return {
    ready,
    close() {
      clearInterval(ticker);
      try { source.close(); } catch (error) { /* already closed */ }
    },
  };
}

function renderCensus(state) {
  const body = $("fit-census-body");
  if (!body) return;
  const order = ["with_leaks", "honest", "baseline"];
  body.innerHTML = order.filter((arm) => state.arms.has(arm)).map((arm) => {
    const data = state.arms.get(arm);
    const rows = data.root_census.map((entry) => `
      <div class="census-row">
        <span class="census-name">${escapeHtml(entry.feature)}</span>
        <span class="census-bar"><i style="width:${(entry.share * 100).toFixed(0)}%"></i></span>
        <span class="census-share">${(entry.share * 100).toFixed(0)}%</span>
      </div>`).join("");
    return `<div class="census-arm ${ARM_KIND[arm] || ""}">
      <h5>${escapeHtml(ARM_TITLE[arm] || arm)}</h5>${rows}</div>`;
  }).join("") || `<p class="note">Waiting for the first fitted model…</p>`;
}

/* The tree is laid out by depth and drawn one node at a time. Nothing about the
   shape is invented: positions come from the node list, which came from
   `estimator.tree_`. */
/* ── the trees, after the fit ────────────────────────────────────────────
   Every tree built during the run is kept, and this is how a reader gets at
   them. Stepping from "every column" to "your confirmed drops removed" on the
   same learner and the same configuration is the comparison the tool exists to
   make, and it is visible in the structure before any score is read. */

function treeGallery(trees) {
  if (!trees?.length) return null;

  const panel = document.createElement("div");
  panel.className = "panel tree-gallery";
  panel.innerHTML = `
    <div class="panel-head">
      <div>
        <h3>The trees your models built</h3>
        <p class="sub">Real structure, read off each fitted estimator. Hover any node
          for its split, its rows and its impurity. Step between arms to watch the
          same learner reach for a different column once the leak is gone.</p>
      </div>
    </div>
    <div class="tree-picker" id="tree-picker"></div>
    <div class="tree-stage">
      <svg id="gallery-tree-svg" viewBox="0 0 420 250" role="img"
           aria-label="a decision tree built during the comparison"></svg>
      <p class="sub" id="gallery-caption"></p>
      <div class="tree-roots" id="gallery-roots"></div>
    </div>`;

  let chosen = 0;
  const draw = () => {
    const data = trees[chosen];
    drawTree(data, panel.querySelector("#gallery-tree-svg"),
             panel.querySelector("#gallery-caption"), { animate: false });
    panel.querySelectorAll(".tree-chip").forEach((chip, index) =>
      chip.classList.toggle("on", index === chosen));
    const roots = panel.querySelector("#gallery-roots");
    roots.innerHTML = (data.root_census || []).slice(0, 5).map((entry) => `
      <span class="root-chip ${ARM_KIND[data.arm] || ""}">
        <b>${escapeHtml(entry.feature)}</b> ${(entry.share * 100).toFixed(0)}%
      </span>`).join("")
      || `<span class="note">No root census for this tree.</span>`;
  };

  panel.querySelector("#tree-picker").innerHTML = trees.map((data, index) => `
    <button type="button" class="tree-chip ${ARM_KIND[data.arm] || ""}" data-index="${index}">
      <span class="chip-arm">${escapeHtml(ARM_TITLE[data.arm] || data.arm)}</span>
      <span class="chip-meta">${escapeHtml(String(data.learner || "").replace(/_/g, " "))}
        · config ${(data.config ?? 0) + 1}</span>
    </button>`).join("");

  panel.querySelector("#tree-picker").addEventListener("click", (event) => {
    const chip = event.target.closest(".tree-chip");
    if (!chip) return;
    chosen = Number(chip.dataset.index);
    draw();
  });

  draw();
  return panel;
}

function treeCaption(data) {
  const config = data.config_detail
    ? Object.entries(data.config_detail)
        .map(([key, value]) => `${key}=${value}`).join(" · ")
    : "";
  return `${ARM_TITLE[data.arm] || data.arm} · ${String(data.learner || "").replace(/_/g, " ")}`
    + ` · depth ${data.depth} · ${data.total_nodes.toLocaleString()} nodes`
    + ` · one of ${data.trees_in_fit}${config ? ` · ${config}` : ""}`;
}

function drawTree(data, svg, caption, { animate = true } = {}) {
  if (!svg || !data.nodes?.length) return;
  svg.innerHTML = "";
  if (caption) caption.textContent = treeCaption(data);

  const depths = [];
  const walk = (index, depth) => {
    if (index === null || index === undefined || index >= data.nodes.length) return;
    (depths[depth] ||= []).push(index);
    walk(data.nodes[index].left, depth + 1);
    walk(data.nodes[index].right, depth + 1);
  };
  walk(0, 0);

  const width = 420, height = 250, padTop = 22, padBottom = 26;
  const levelHeight = (height - padTop - padBottom) / Math.max(depths.length - 1, 1);
  const at = [];
  depths.forEach((level, depth) => {
    level.forEach((index, position) => {
      at[index] = {
        x: ((position + 1) / (level.length + 1)) * width,
        y: padTop + depth * levelHeight,
      };
    });
  });

  const kind = ARM_KIND[data.arm] || "leaked";
  const color = { leaked: "#b3261e", honest: "#2563a8", baseline: "#5f7a1f" }[kind];
  const depthOf = [];
  depths.forEach((level, depth) => level.forEach((index) => { depthOf[index] = depth; }));
  const pieces = [];
  data.nodes.forEach((node, index) => {
    const here = at[index];
    if (!here) return;
    for (const child of [node.left, node.right]) {
      if (child === null || !at[child]) continue;
      pieces.push({ kind: "edge", from: here, to: at[child] });
    }
  });
  data.nodes.forEach((node, index) => {
    if (at[index]) pieces.push({ kind: "node", node, at: at[index], index });
  });

  const biggest = Math.max(...data.nodes.map((node) => node.samples)) || 1;
  pieces.forEach((piece, order) => {
    const delay = animate ? order * 14 : 0;
    if (piece.kind === "edge") {
      const line = make("line", { x1: piece.from.x, y1: piece.from.y,
                                  x2: piece.to.x, y2: piece.to.y,
                                  stroke: color, "stroke-width": 1, "stroke-opacity": 0.35 });
      line.style.opacity = 0;
      svg.appendChild(line);
      setTimeout(() => { line.style.transition = "opacity 220ms"; line.style.opacity = 1; }, delay);
      return;
    }
    const radius = 2.5 + 5 * Math.sqrt(piece.node.samples / biggest);
    const circle = make("circle", { cx: piece.at.x, cy: piece.at.y, r: radius,
                                    fill: piece.node.feature ? color : "#c9cec2",
                                    "fill-opacity": piece.node.feature ? 0.85 : 0.6 });
    // Every node says what it did, on hover. A leaf says so rather than
    // showing an empty tooltip.
    const explain = make("title");
    explain.textContent = piece.node.feature
      ? `${piece.node.feature} \u2264 ${piece.node.threshold}`
        + ` \u00b7 ${piece.node.samples.toLocaleString()} rows`
        + ` \u00b7 impurity ${piece.node.impurity}`
      : `leaf \u00b7 ${piece.node.samples.toLocaleString()} rows`
        + ` \u00b7 impurity ${piece.node.impurity}`;
    circle.appendChild(explain);
    circle.style.opacity = 0;
    circle.style.transform = "scale(0.4)";
    circle.style.transformOrigin = `${piece.at.x}px ${piece.at.y}px`;
    svg.appendChild(circle);
    setTimeout(() => {
      circle.style.transition = "opacity 200ms, transform 260ms cubic-bezier(0.34,1.56,0.64,1)";
      circle.style.opacity = 1;
      circle.style.transform = "scale(1)";
    }, delay);

    // The first two levels are named on the face of the chart. Deeper than
    // that the labels collide and stop being readable, and the argument is
    // carried by the top of the tree anyway: the root is the column the model
    // reached for before anything else.
    const level = depthOf[piece.index] ?? 99;
    if (level <= 1 && piece.node.feature) {
      const isRoot = piece.index === 0;
      const label = make("text", {
        x: piece.at.x,
        y: isRoot ? piece.at.y - radius - 7 : piece.at.y - radius - 5,
        "text-anchor": "middle",
        class: isRoot ? "tree-root-label" : "tree-node-label",
      });
      label.textContent = isRoot
        ? piece.node.feature
        : truncate(piece.node.feature, 14);
      label.style.opacity = 0;
      svg.appendChild(label);
      setTimeout(() => {
        label.style.transition = "opacity 260ms";
        label.style.opacity = 1;
      }, delay + 120);
    }
  });
}

/* A server that fails outside its own error handling answers with plain text,
   not JSON. Calling `.json()` on that throws a parser message with no bearing
   on what went wrong: Safari says "The string did not match the expected
   pattern", which sent this exact bug on a long detour. Read the body once,
   then decide what it is. */
/* Ask for the answer until it exists.

   A 202 means the fit is still going, which on a small host can be several
   minutes; the visuals are driven by the event stream in the meantime, so this
   only has to be patient. Every failure the server can describe arrives as a
   normal error response and is raised. */
async function collectImpact(url, signal) {
  const started = Date.now();
  for (;;) {
    if (signal.aborted) throw Object.assign(new Error("stopped"), { name: "AbortError" });
    const response = await fetch(url, { signal, cache: "no-store" });
    if (response.status === 202) {
      // Give up long after any real fit would have finished, so a server that
      // died mid-run does not leave the page waiting forever.
      if (Date.now() - started > 30 * 60 * 1000) {
        throw new Error("the comparison did not finish within thirty minutes");
      }
      await sleep(2000);
      continue;
    }
    const body = await readResult(response);
    if (!response.ok) throw new Error(body.detail || "the comparison failed");
    return body;
  }
}

async function readResult(response) {
  const body = await response.text();
  try {
    return JSON.parse(body);
  } catch (error) {
    const snippet = body.trim().slice(0, 200) || `HTTP ${response.status}`;
    return { detail: `The server did not return a result. It said: ${snippet}` };
  }
}

/* Browsers report DOM errors as bare sentences. "The string did not match the
   expected pattern" names no file, no line and no value, which makes it a
   report nobody can act on. Anything thrown while measuring is captured with
   its origin so the next occurrence is diagnosable in one look rather than one
   afternoon. */
let lastScriptError = null;

window.addEventListener("error", (event) => {
  if (!event.filename) return;
  lastScriptError = {
    message: event.message,
    where: `${event.filename.split("/").pop()}:${event.lineno}:${event.colno}`,
    stack: event.error?.stack,
  };
});

window.addEventListener("unhandledrejection", (event) => {
  lastScriptError = {
    message: String(event.reason?.message || event.reason),
    where: "an unawaited promise",
    stack: event.reason?.stack,
  };
});

/* Where a failure came from, preferring the exception handed in over anything
   the window handler happened to catch. A stack line naming this file is worth
   more than the top frame, which is usually inside the browser's own plumbing. */
function errorDetail(thrown) {
  const stack = thrown?.stack || lastScriptError?.stack;
  if (!stack) return "";
  const line = stack.split("\n")
    .map((entry) => entry.trim())
    .find((entry) => entry.includes("app.js")) || stack.split("\n")[0];
  return line ? `<span class="fail-where">${escapeHtml(line)}</span>` : "";
}

function failImpact(message, thrown = null) {
  // Also to the console, because a stack is easier to read there and a person
  // reporting this can paste it.
  if (thrown) console.error("Crucible: the comparison failed", thrown);
  status("failed", "bad");
  const detail = errorDetail(thrown);
  $("results").innerHTML = `
    <div class="panel">
      <div class="banner">
        <b>The comparison did not finish.</b>
        <span>${escapeHtml(message)}</span>
        ${detail}
        <span class="fail-next">Nothing has been changed. Adjust the column set below
          and measure again, or reload if this persists.</span>
      </div>
    </div>`;
  updateEditorFoot();
}

/* ══════════ STAGE 6 · RESULTS ══════════ */

function finishImpact(result) {
  app.impact = result;
  status("complete", "go");
  const groups = result.group_column
    ? `Folds are grouped on ${result.group_column}, so no unit appears on both sides of a split.`
    : "Folds are stratified on the target.";
  $("results-protocol").textContent =
    `${result.n_rows_used.toLocaleString()} of ${result.n_rows_total.toLocaleString()} rows, ` +
    `${result.n_classes} classes (${result.class_labels.join(", ")}). ${groups}`;

  $("results").innerHTML = sameArmsNotice(result) + verdictBanner(result);
  Object.entries(result.learners).forEach(([name, arms]) => {
    $("results").appendChild(learnerBlock(name, arms, result));
  });
  // The trees are gathered while the fit runs and outlive it, because during
  // the run only the newest is on screen and the comparison between arms needs
  // all of them.
  const gallery = treeGallery(app.fitTrees);
  if (gallery) $("results").appendChild(gallery);
  measuredSet = [...result.drop_list].sort();
  renderEditor();
  renderResultTable();
}

const LEARNER_NAME = { random_forest: "Random forest", gradient_boosting: "Gradient boosting" };

/* When two arms hold the same columns they are one fit reported twice, and
   every difference between them is exactly zero. A zero read as a measurement
   says "there was no leakage here"; the truth is that nothing was removed, and
   the two claims could not be further apart. Said first, above the numbers,
   because a reader who has already looked at four identical figures has drawn
   the wrong conclusion by the time they reach an explanation. */
const collapsed = (result, left, right) =>
  (result.identical_arms || []).some(({ arms }) =>
    arms.includes(left) && arms.includes(right));

function sameArmsNotice(result) {
  const pairs = result.identical_arms || [];
  const ignored = result.drops_ignored || [];
  if (!pairs.length) return "";

  const lines = [];
  if (collapsed(result, "with_leaks", "honest")) {
    lines.push(`<p class="warn-line"><b>This is not a comparison.</b> Both arms were fit on
      the same columns, so the numbers below are one model reported twice and every
      difference between them is zero by construction, not by measurement.</p>
      <p class="warn-sub">${ignored.length
        ? `Nothing you confirmed is a feature column of this table: ` +
          `${ignored.map(escapeHtml).join(", ")}. The target and any grouping column are ` +
          `already held out, so removing them changes nothing.`
        : `Your confirmed drops encode to no columns, so removing them left the feature ` +
          `set unchanged.`} Confirm a column that is in the table and measure again.</p>`);
  }
  if (collapsed(result, "honest", "baseline")) {
    lines.push(`<p class="warn-line">A correlation threshold would have removed exactly the
      columns you confirmed, so the cleaned and correlation arms are one fit. On this table
      the cheap screen and the semantic one agree.</p>`);
  }
  if (collapsed(result, "with_leaks", "baseline")) {
    lines.push(`<p class="warn-line">The correlation threshold removed nothing here, so its
      arm is the with-leaks arm under another name. Every leak in this table sits below the
      cutoff a statistic could use.</p>`);
  }
  if (!lines.length) return "";
  const severe = collapsed(result, "with_leaks", "honest");
  return `<div class="arm-warn${severe ? " severe" : ""}">
      <span class="verdict-eyebrow">${severe ? "Read this first" : "Worth knowing"}</span>
      ${lines.join("")}
    </div>`;
}

/* The one sentence a reader should leave with, before any table.

   Four cards of numbers falling from 0.97 to 0.79, every arrow pointing down,
   reads at a glance as "the tool made things worse". It is the opposite, and
   saying so in words costs one line. */
function verdictBanner(result) {
  // A difference of zero between one fit and itself is not a finding, and the
  // notice above has already said so.
  if (collapsed(result, "with_leaks", "honest")) return "";
  const gaps = Object.values(result.learners)
    .map((arms) => arms.inflation.macro_f1)
    .filter((value) => value !== null && value !== undefined);
  if (!gaps.length) return "";
  const worst = Math.max(...gaps);
  const dropped = result.drop_list || [];
  const noun = dropped.length === 1 ? "column was" : "columns were";
  return `<div class="verdict">
      <span class="verdict-eyebrow">What the leakage was worth</span>
      <p class="verdict-line">Your model scored
        <b>${worst.toFixed(3)} macro F1 higher</b> than it deserved to.</p>
      <p class="verdict-sub">${dropped.length} ${noun} reading the answer rather than
        predicting it${dropped.length ? `: ${dropped.map(escapeHtml).join(", ")}` : ""}.
        Everything below compares the same models fit with and without them.</p>
    </div>`;
}

function learnerBlock(name, arms, result = {}) {
  const block = document.createElement("div");
  block.className = "learner";
  const leaked = arms.with_leaks, honest = arms.honest, gap = arms.inflation;
  const oneFit = collapsed(result, "with_leaks", "honest");
  if (oneFit) block.classList.add("one-fit");

  block.innerHTML = `
    <div class="learner-head">
      <h3>${LEARNER_NAME[name] || name}</h3>
      <span class="note">${leaked.n_features} encoded features with leaks · ${honest.n_features} without</span>
    </div>
    ${oneFit ? `<p class="one-fit-strip">Both columns below are the same fit. The
      difference is zero because the arms are identical, not because the columns were
      harmless.</p>` : ""}

    <div class="headline">
      ${headlineCard("Macro F1", leaked.macro.f1, honest.macro.f1, gap.macro_f1)}
      ${headlineCard("Weighted F1", leaked.weighted.f1, honest.weighted.f1, gap.weighted_f1)}
      ${headlineCard("Accuracy", leaked.accuracy, honest.accuracy, gap.accuracy)}
      ${headlineCard("AUC", leaked.auc, honest.auc, gap.auc)}
    </div>

    ${baselineBanner(arms)}

    <div class="legend">
      <span><i class="leaked"></i>with leaks</span>
      <span><i class="honest"></i>cleaned</span>
      <span>${leaked.n_classes === 2 ? "each arm at its own best-F1 threshold" : "predicted class is the highest probability"}</span>
    </div>

    <h4 class="matrix-title">Where the mistakes actually land</h4>
    <p class="sub">Rows are what the row really was; columns are what the model predicted. The
      diagonal is correct and everything off it is a mistake. An averaged score hides which kind of
      mistake got worse — this does not.</p>
    <div class="matrices">
      ${matrixCard("With leaks", "leaked", leaked)}
      ${matrixCard("Cleaned", "honest", honest, leaked)}
    </div>

    ${perClassTable(leaked, honest)}`;

  const charts = document.createElement("div");
  charts.className = "small-charts";
  const classes = curveClasses(arms);
  charts.appendChild(chartCard("ROC",
    "How well each fit separates the classes at every cutoff. Higher and further "
    + "left is better; the diagonal is chance.",
    (index) => rocSvg(arms, index), classes));
  charts.appendChild(chartCard("Precision–recall",
    "How precision holds up as more recall is demanded. More informative than ROC "
    + "when one class is rare.",
    (index) => prSvg(arms, index), classes));
  if (leaked.sweep && sweepSvg(arms)) {
    charts.appendChild(chartCard("F1 across thresholds",
      "F1 at every cutoff. The marked line on each curve is the threshold actually "
      + "used, chosen inside the training folds.",
      () => sweepSvg(arms)));
  }
  block.appendChild(charts);
  block.appendChild(foldStrip(arms));
  return block;
}

/* The third arm answers the question a reader should be asking: would a cheap
   correlation threshold have done this job? On Titanic it does not — it deletes
   `sex` and keeps both leaks, so its "cleaned" score stays inflated. When that
   happens it is worth saying loudly; when the baseline does just as well, that
   is worth saying too. */
function baselineBanner(arms) {
  const baseline = arms.baseline;
  if (!baseline) return "";
  const residual = arms.baseline_residual?.macro_f1;
  const honest = arms.honest.macro.f1;
  const stillInflated = residual !== null && residual !== undefined && residual > 0.02;
  return `<div class="baseline-arm ${stillInflated ? "bad" : "ok"}">
      <span class="ba-label">If you had used a correlation threshold instead</span>
      <div class="ba-row">
        <span class="ba-score">${fixed(baseline.macro.f1, 3)}</span>
        <span class="ba-vs">macro F1, against <b>${fixed(honest, 3)}</b> for the set you confirmed</span>
      </div>
      <p class="ba-note">${stillInflated
        ? `Still <b>${signed(residual, 3)}</b> above the honest score. The threshold ` +
          `removed the wrong columns and left the leak in, so it bought no protection at all.`
        : `Within ${fixed(Math.abs(residual || 0), 3)} of the honest score, so on this ` +
          `table the cheap check would have been enough. Worth knowing.`}</p>
    </div>`;
}

function headlineCard(label, leaked, honest, gap) {
  return `<div class="hl">
      <span>${label}</span>
      <div class="hl-pair">
        <span class="hl-leaked">${fixed(leaked, 3)}</span>
        <span class="hl-arrow">→</span>
        <span class="hl-honest">${fixed(honest, 3)}</span>
      </div>
      <span class="hl-gap">${signed(gap, 3)} overstated by the leaks</span>
    </div>`;
}

/* The confusion matrix: a full grid with the axes named in the corner, one
   header card per class on each edge, and a flat two-color fill. Shading by
   magnitude was tried and discarded — it turns every cell a different tint and
   makes the diagonal harder to find, not easier. */
function matrixCard(title, tone, arm, compareWith = null) {
  const labels = arm.labels;
  const size = labels.length;
  const errors = arm.confusion.reduce((total, row, r) =>
    total + row.reduce((sum, cell, c) => sum + (r === c ? 0 : cell), 0), 0);

  const columns = `grid-template-columns: minmax(96px, 0.9fr) repeat(${size}, minmax(96px, 1fr))`;
  let cells = `<div class="mx-corner">actual ╲ predicted</div>`;
  labels.forEach((label) =>
    cells += `<div class="mx-head"><small>predicted</small>${escapeHtml(label)}</div>`);

  arm.confusion.forEach((row, rowIndex) => {
    cells += `<div class="mx-head"><small>actual</small>${escapeHtml(labels[rowIndex])}</div>`;
    const rowTotal = row.reduce((total, cell) => total + cell, 0);
    row.forEach((cell, columnIndex) => {
      const correct = rowIndex === columnIndex;
      const share = rowTotal ? (cell / rowTotal) * 100 : 0;
      cells += `<div class="mx-cell ${correct ? "correct" : "mistake"}">
          <span class="mx-tag">${correct ? "correct" : "mistake"}</span>
          <strong>${cell.toLocaleString()}</strong>
          <small>${share.toFixed(1)}% of actual ${escapeHtml(labels[rowIndex])}</small>
        </div>`;
    });
  });

  let footer = `<span>total mistakes</span><b>${errors.toLocaleString()}</b>`;
  if (compareWith) {
    const otherErrors = compareWith.confusion.reduce((total, row, r) =>
      total + row.reduce((sum, cell, c) => sum + (r === c ? 0 : cell), 0), 0);
    const difference = errors - otherErrors;
    footer = `<span>total mistakes</span><b class="${difference > 0 ? "worse" : ""}">` +
      `${errors.toLocaleString()}${difference > 0 ? ` · ${difference.toLocaleString()} more` : ""}</b>`;
  }

  const threshold = arm.threshold !== undefined ? ` at threshold ${arm.threshold}` : "";
  return `<div class="matrix-card ${tone}">
      <h5><i></i>${title}</h5>
      <span class="note">${arm.config ? Object.entries(arm.config).map(([k, v]) => `${k}=${v}`).join(" · ") : ""}${threshold}</span>
      <div class="matrix" style="${columns}">${cells}</div>
      <div class="mx-total">${footer}</div>
    </div>`;
}

function perClassTable(leaked, honest) {
  const rows = leaked.per_class.map((entry, index) => {
    const other = honest.per_class[index];
    return `<tr>
        <td>${escapeHtml(entry.label)}</td>
        <td class="num">${entry.support.toLocaleString()}</td>
        <td class="num">${fixed(entry.precision, 3)}</td>
        <td class="num">${fixed(other.precision, 3)}</td>
        <td class="num">${fixed(entry.recall, 3)}</td>
        <td class="num">${fixed(other.recall, 3)}</td>
        <td class="num">${fixed(entry.f1, 3)}</td>
        <td class="num">${fixed(other.f1, 3)}</td>
      </tr>`;
  }).join("");
  return `<table class="class-table">
      <thead><tr>
        <th>class</th><th>rows</th>
        <th>precision leaked</th><th>precision clean</th>
        <th>recall leaked</th><th>recall clean</th>
        <th>F1 leaked</th><th>F1 clean</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/* One chart, and — where the target has more than two classes — the switch
   between them.

   Nine curves in one frame is not a figure, it is a thicket: three arms times
   three classes, every ROC of them ending at the same corner. So a multiclass
   chart draws one class at a time, three curves, one per arm, which is the
   shape this comparison actually has. The class being shown is drawn into the
   figure itself rather than sitting in the page around it, so an exported SVG
   still says what it is. */
function chartCard(title, explanation, build, classes = null) {
  const card = document.createElement("div");
  card.className = "sc";
  const faceted = Array.isArray(classes) && classes.length > 1;
  card.innerHTML = `<div class="sc-title">${title}</div>
    <p class="sc-sub">${explanation}${faceted
      ? " Each class is scored against all the others; pick which one below."
      : ""}</p>`;

  let index = 0;
  let svg = build(index);
  const holder = document.createElement("div");
  holder.className = "sc-plot";
  holder.appendChild(svg);

  if (faceted) {
    const chips = document.createElement("div");
    chips.className = "sc-classes";
    chips.innerHTML = `<span class="sc-classes-label">class</span>` +
      classes.map((label, position) =>
        `<button type="button" class="sc-class${position ? "" : " on"}"
           data-index="${position}">${escapeHtml(label)}</button>`).join("");
    chips.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      index = Number(button.dataset.index);
      chips.querySelectorAll("button").forEach((other) =>
        other.classList.toggle("on", other === button));
      const next = build(index);
      holder.replaceChild(next, svg);
      svg = next;
    });
    card.appendChild(chips);
  }
  card.appendChild(holder);

  // A figure that cannot leave the page is not much use to somebody writing a
  // paper. Vector, not a screenshot, and styled standalone so it opens the same
  // way anywhere.
  const foot = document.createElement("div");
  foot.className = "sc-foot";
  foot.innerHTML = `<button type="button" class="sc-export">Download SVG</button>
    <span class="sc-hint">Hover to read values · click a legend entry to hide a series</span>`;
  foot.querySelector(".sc-export").addEventListener("click", () =>
    exportSvg(svg, faceted ? `${title} ${classes[index]}` : title));
  card.appendChild(foot);
  return card;
}

/* The downloaded file carries its own stylesheet, because the page's lives in
   a stylesheet the file does not travel with. Kept in step with the `.sc`
   rules: a figure that looks one way on screen and another in the paper is
   worse than no export at all. */
const FIGURE_STYLE = `
  svg { background: #fffdf7; }
  text { font-family: "DM Mono", ui-monospace, Menlo, monospace; fill: #17201a; }
  .axis { font-size: 8px; fill: #6c7669; }
  .axis-title { font-size: 8.5px; fill: #4f5a4c; }
  .plot-header { font-size: 9px; font-weight: 650; fill: #17201a; }
  .grid-line { stroke: rgba(23,32,25,0.10); stroke-width: 0.6; }
  .spine, .tick-mark { stroke: rgba(23,32,25,0.28); stroke-width: 0.8; }
  .reference { stroke: rgba(23,32,25,0.28); stroke-width: 0.8; stroke-dasharray: 3 3; }
  .curve { fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
  .legend-box { fill: #fffdf7; stroke: rgba(23,32,25,0.16); }
  .legend-text { font-size: 7.4px; font-weight: 500; }
  .legend-heading { font-size: 6.6px; fill: #6c7669; letter-spacing: 0.09em; }
  .plot-note { font-size: 7.4px; fill: #4f5a4c;
               paint-order: stroke; stroke: #fffdf7; stroke-width: 3px; stroke-linejoin: round; }
  .curve-label { font-size: 7.4px; }
  .hover-layer, .legend-hit { display: none; }`;

function exportSvg(svg, title) {
  const copy = svg.cloneNode(true);
  copy.setAttribute("xmlns", SVG_NS);
  copy.querySelectorAll(".hover-layer").forEach((node) => node.remove());
  const style = document.createElementNS(SVG_NS, "style");
  style.textContent = FIGURE_STYLE;
  copy.insertBefore(style, copy.firstChild);
  const blob = new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n${copy.outerHTML}`],
                        { type: "image/svg+xml" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `crucible-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.svg`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

/* Reading a value off a printed curve means holding a ruler to it. Here the
   chart says what it is: a rule follows the pointer and every series reports
   its value at that x. Clicking a legend entry hides its series, which is the
   only way to see a curve that another one is sitting exactly on top of. */
function makeInteractive(svg, inner, { xLabel = "x", yLabel = "y", xOf = null } = {}) {
  if (!svg._series?.length) return svg;

  const layer = make("g", { class: "hover-layer" });
  const rule = make("line", { class: "hover-rule", y1: PLOT.top, y2: inner.y0 });
  const box = make("rect", { class: "hover-box", rx: 4 });
  const lines = svg._series.map(() => make("text", { class: "hover-text" }));
  const heading = make("text", { class: "hover-heading" });
  layer.append(rule, box, heading, ...lines);
  svg.appendChild(layer);

  const show = (on) => { layer.style.display = on ? "block" : "none"; };
  show(false);

  svg.addEventListener("pointerleave", () => show(false));
  svg.addEventListener("pointermove", (event) => {
    const bounds = svg.getBoundingClientRect();
    const scale = PLOT.width / bounds.width;
    const x = (event.clientX - bounds.left) * scale;
    if (x < inner.x0 || x > inner.x0 + inner.w) return show(false);
    show(true);

    const fraction = (x - inner.x0) / inner.w;
    rule.setAttribute("x1", x);
    rule.setAttribute("x2", x);
    heading.textContent = `${xLabel} ${(xOf ? xOf(fraction) : fraction).toFixed(3)}`;

    const readings = svg._series.map((series, index) => {
      if (series.hidden) return null;
      // Nearest point by x. Curves are dense enough that interpolating would
      // add precision the data does not have.
      let best = series.points[0], gap = Infinity;
      for (const point of series.points) {
        const distance = Math.abs(point[0] - fraction);
        if (distance < gap) { gap = distance; best = point; }
      }
      return { index, label: series.label, value: best[1], color: series.color };
    }).filter(Boolean);

    const left = fraction > 0.55;
    const boxX = left ? inner.x0 + 6 : inner.x0 + inner.w - 96;
    const boxY = PLOT.top + 4;
    box.setAttribute("x", boxX); box.setAttribute("y", boxY);
    box.setAttribute("width", 92);
    box.setAttribute("height", 13 + readings.length * 10);
    heading.setAttribute("x", boxX + 6); heading.setAttribute("y", boxY + 10);

    lines.forEach((line, index) => {
      const reading = readings[index];
      if (!reading) { line.textContent = ""; return; }
      line.setAttribute("x", boxX + 6);
      line.setAttribute("y", boxY + 21 + index * 10);
      line.setAttribute("fill", reading.color);
      line.textContent = `${reading.label} ${reading.value.toFixed(3)}`;
    });
  });
  return svg;
}

/* ── charts ───────────────────────────────────────────────────────────

   Three rules the earlier version broke.

   Color is never the only channel. The previous palette separated the two
   arms by ΔE 4.0 under protanopia — indistinguishable — so every curve now
   carries a dash pattern and a label sitting at its own end.

   Every class gets its own curve, labeled. Drawing three classes in one
   color, as this did, makes a multi-class chart unreadable at exactly the
   moment it matters.

   The operating point is drawn. Choosing a threshold carefully and then
   plotting F1 across every threshold without marking the one in use omits
   the only decision the chart exists to support.                            */

const SERIES = {
  leaked:   { color: "#b3261e", dash: "",      label: "with leaks" },
  honest:   { color: "#2563a8", dash: "6 3",   label: "cleaned" },
  baseline: { color: "#5f7a1f", dash: "2 3",   label: "correlation" },
};

// `right` was 54 to make room for labels stuck on the end of each curve.
// Those collided (every ROC curve ends at 1,1) and are now a legend, so the
// gutter goes back to the plot.
// Close to square, because a ROC read on a wide flat box exaggerates every
// difference along one axis and flattens it along the other. `top` carries the
// figure header — the class a panel is drawn for — which is inside the exported
// file rather than in the page around it.
const PLOT = { width: 320, height: 258, left: 44, right: 14, top: 26, bottom: 38 };
const SVG_NS = "http://www.w3.org/2000/svg";
const make = (tag, attributes = {}) => {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
};

function plotFrame(xLabel, yLabel, xTicks = [0, 0.25, 0.5, 0.75, 1], header = null) {
  const label = header ? `${yLabel} against ${xLabel}, ${header}`
                       : `${yLabel} against ${xLabel}`;
  const svg = make("svg", { viewBox: `0 0 ${PLOT.width} ${PLOT.height}`,
                            role: "img", "aria-label": label });
  const titleNode = make("title");
  titleNode.textContent = label;
  svg.appendChild(titleNode);
  const inner = {
    x0: PLOT.left, y0: PLOT.height - PLOT.bottom,
    w: PLOT.width - PLOT.left - PLOT.right, h: PLOT.height - PLOT.top - PLOT.bottom,
  };
  const at = (x, y) => [inner.x0 + x * inner.w, inner.y0 - y * inner.h];

  for (const value of [0, 0.25, 0.5, 0.75, 1]) {
    const [, y] = at(0, value);
    svg.appendChild(make("line", { class: "grid-line", x1: inner.x0, y1: y,
                                   x2: inner.x0 + inner.w, y2: y }));
    const tick = make("text", { class: "axis", x: inner.x0 - 7, y: y + 3,
                                "text-anchor": "end" });
    tick.textContent = value === 0 || value === 1 ? String(value) : value.toFixed(2).slice(1);
    svg.appendChild(tick);
  }
  for (const value of xTicks) {
    const [x] = at(value, 0);
    svg.appendChild(make("line", { class: "tick-mark", x1: x, y1: inner.y0,
                                   x2: x, y2: inner.y0 + 3.5 }));
    const tick = make("text", { class: "axis", x, y: inner.y0 + 15,
                                "text-anchor": "middle" });
    tick.textContent = value === 0 || value === 1 ? String(value) : value.toFixed(2).slice(1);
    svg.appendChild(tick);
  }
  // Spines. A plot area bounded on two sides reads as a figure; one bounded by
  // nothing but its grid reads as a sketch.
  svg.appendChild(make("line", { class: "spine", x1: inner.x0, y1: PLOT.top,
                                 x2: inner.x0, y2: inner.y0 }));
  svg.appendChild(make("line", { class: "spine", x1: inner.x0, y1: inner.y0,
                                 x2: inner.x0 + inner.w, y2: inner.y0 }));

  const xTitle = make("text", { class: "axis-title", x: inner.x0 + inner.w / 2,
                                y: PLOT.height - 6, "text-anchor": "middle" });
  xTitle.textContent = xLabel;
  svg.appendChild(xTitle);
  const yTitle = make("text", { class: "axis-title", "text-anchor": "middle",
    transform: `translate(12,${PLOT.top + inner.h / 2}) rotate(-90)` });
  yTitle.textContent = yLabel;
  svg.appendChild(yTitle);

  if (header) {
    const caption = make("text", { class: "plot-header", x: inner.x0, y: PLOT.top - 10 });
    caption.textContent = header;
    svg.appendChild(caption);
  }
  return { svg, inner, at };
}

/* A legend inside the plot, in a corner the curves do not reach: ROC keeps its
   bottom-right empty below the diagonal, precision-recall and the F1 sweep keep
   their bottom-left empty. This replaces labels pinned to the end of each
   curve, which stacked on top of each other because every ROC curve ends at the
   same point. */
function plotLegend(svg, inner, entries, corner = "br", heading = null) {
  if (!entries.length) return;
  const lineHeight = 13;
  const widest = Math.max(...entries.map((e) =>
    (e.label + (e.value ? ` · ${e.value}` : "")).length));
  const boxWidth = Math.max(widest * 4.35 + 30, heading ? 62 : 0);
  const headRoom = heading ? 11 : 0;
  const boxHeight = entries.length * lineHeight + 9 + headRoom;
  const x = corner.includes("r") ? inner.x0 + inner.w - boxWidth - 4 : inner.x0 + 6;
  const y = corner.includes("b") ? inner.y0 - boxHeight - 4 : PLOT.top + 4;

  svg.appendChild(make("rect", { class: "legend-box", x, y,
                                 width: boxWidth, height: boxHeight, rx: 5 }));
  if (heading) {
    const cap = make("text", { class: "legend-heading", x: x + 7, y: y + 10 });
    cap.textContent = heading;
    svg.appendChild(cap);
  }
  entries.forEach((entry, index) => {
    const lineY = y + 9 + headRoom + index * lineHeight;
    const swatch = make("line", { x1: x + 7, y1: lineY, x2: x + 21, y2: lineY,
                                  stroke: SERIES[entry.kind].color,
                                  "stroke-width": 2.2, "stroke-linecap": "round" });
    if (SERIES[entry.kind].dash) {
      swatch.setAttribute("stroke-dasharray", SERIES[entry.kind].dash);
    }
    svg.appendChild(swatch);
    const text = make("text", { class: "legend-text", x: x + 26, y: lineY + 3.2 });
    text.textContent = entry.value ? `${entry.label} · ${entry.value}` : entry.label;
    svg.appendChild(text);

    // A hit area over the row, because 7px type is not a click target.
    const hit = make("rect", { class: "legend-hit", x: x + 2, y: lineY - 6,
                               width: boxWidth - 4, height: lineHeight });
    hit.addEventListener("click", () => {
      const series = (svg._series || []).filter((item) => item.kind === entry.kind);
      const hiding = !series[0]?.hidden;
      series.forEach((item) => {
        item.hidden = hiding;
        item.element.style.opacity = hiding ? 0.12 : 1;
      });
      swatch.style.opacity = hiding ? 0.3 : 1;
      text.style.opacity = hiding ? 0.4 : 1;
    });
    svg.appendChild(hit);
  });
}

/* Nothing is written at the end of a curve any more. Every ROC curve ends at
   the same corner and every precision-recall curve at the same edge, so labels
   pinned there stack into a block of text sitting on top of the data. The
   legend carries the names, and one panel per class keeps the count of names
   down to the number of arms. */
function drawCurve(svg, inner, points, kind, { label = null, animate = true } = {}) {
  if (!points?.length) return null;
  const spec = SERIES[kind];
  const path = points.map(([x, y], index) =>
    `${index ? "L" : "M"}${(inner.x0 + x * inner.w).toFixed(1)},${(inner.y0 - y * inner.h).toFixed(1)}`
  ).join("");
  const element = make("path", { class: "curve", d: path, stroke: spec.color });
  if (spec.dash) element.setAttribute("stroke-dasharray", spec.dash);
  svg.appendChild(element);
  (svg._series ||= []).push({
    kind, label: label || spec.label, color: spec.color, points, element,
  });

  if (animate && !spec.dash) {
    const length = element.getTotalLength?.() || 600;
    element.style.strokeDasharray = length;
    element.style.strokeDashoffset = length;
    requestAnimationFrame(() => {
      element.style.transition = "stroke-dashoffset 900ms cubic-bezier(0.32,0.72,0,1)";
      element.style.strokeDashoffset = 0;
    });
  }
  return element;
}

/* Which arms exist, in a fixed order so color follows the arm and never its
   rank — a filter that removes one arm must not repaint the others. */
const armsOf = (arms) => [
  ["leaked", arms.with_leaks], ["honest", arms.honest], ["baseline", arms.baseline],
].filter(([, arm]) => arm);

/* The class names a faceted chart switches between, or null when there is only
   one curve to draw and nothing to switch. Read off the curves rather than off
   `class_labels`, because a class that never appears in a held-out fold has no
   curve and must not get a chip that draws an empty panel. */
const curveClasses = (arms) =>
  arms.with_leaks.n_classes > 2
    ? arms.with_leaks.roc_curves.map((curve) => curve.label)
    : null;

function rocSvg(arms, classIndex = 0) {
  const multiclass = arms.with_leaks.n_classes > 2;
  const header = multiclass
    ? `${arms.with_leaks.roc_curves[classIndex]?.label ?? ""} against the rest` : null;
  const { svg, inner } = plotFrame("false positive rate", "true positive rate",
                                   [0, 0.25, 0.5, 0.75, 1], header);
  svg.appendChild(make("line", { class: "reference", x1: inner.x0, y1: inner.y0,
                                 x2: inner.x0 + inner.w, y2: inner.y0 - inner.h }));
  const entries = [];
  for (const [kind, arm] of armsOf(arms)) {
    const curve = arm.roc_curves[classIndex];
    if (!curve) continue;
    drawCurve(svg, inner, curve.points, kind, { label: SERIES[kind].label });
    // Per class where there are classes to choose between, and the headline
    // figure otherwise. An averaged number cannot label a single class's curve.
    const area = multiclass ? curve.auc : (curve.auc ?? arm.auc);
    entries.push({ kind, label: SERIES[kind].label,
                   value: area === undefined || area === null ? null : fixed(area, 3) });
  }
  plotLegend(svg, inner, entries, "br", "AUC");
  return makeInteractive(svg, inner,
                         { xLabel: "false positive rate", yLabel: "true positive rate" });
}

function prSvg(arms, classIndex = 0) {
  const multiclass = arms.with_leaks.n_classes > 2;
  const shown = arms.with_leaks.pr_curves[classIndex];
  const header = multiclass ? `${shown?.label ?? ""} against the rest` : null;
  const { svg, inner } = plotFrame("recall", "precision", [0, 0.25, 0.5, 0.75, 1], header);

  const base = shown?.baseline;
  if (base !== undefined) {
    const y = inner.y0 - base * inner.h;
    svg.appendChild(make("line", { class: "reference", x1: inner.x0, y1: y,
                                   x2: inner.x0 + inner.w, y2: y }));
    // Left-aligned and haloed. Pinned to the right edge it sat under whichever
    // curve happened to pass through that corner, which on a good fit is all
    // of them.
    const note = make("text", { class: "plot-note", x: inner.x0 + 4, y: y - 5 });
    note.textContent = `base rate ${base.toFixed(3)}`;
    svg.appendChild(note);
  }
  const entries = [];
  for (const [kind, arm] of armsOf(arms)) {
    const curve = arm.pr_curves[classIndex];
    if (!curve) continue;
    drawCurve(svg, inner, curve.points, kind, { label: SERIES[kind].label });
    const precision = multiclass ? curve.ap : (curve.ap ?? arm.average_precision);
    entries.push({ kind, label: SERIES[kind].label,
                   value: precision === undefined || precision === null
                     ? null : fixed(precision, 3) });
  }
  plotLegend(svg, inner, entries, "bl", "AP");
  return makeInteractive(svg, inner, { xLabel: "recall", yLabel: "precision" });
}

function sweepSvg(arms) {
  const all = armsOf(arms).filter(([, arm]) => arm.sweep);
  if (!all.length) return null;
  const thresholds = all.flatMap(([, arm]) => arm.sweep.map(([t]) => t));
  const low = Math.min(...thresholds), high = Math.max(...thresholds);
  const span = high - low || 1;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => low + f * span);
  const { svg, inner } = plotFrame("decision threshold", "F1", []);

  for (const value of ticks) {
    const x = inner.x0 + ((value - low) / span) * inner.w;
    svg.appendChild(make("line", { class: "tick-mark", x1: x, y1: inner.y0,
                                   x2: x, y2: inner.y0 + 3.5 }));
    const tick = make("text", { class: "axis", x, y: inner.y0 + 15, "text-anchor": "middle" });
    tick.textContent = value.toFixed(2);
    svg.appendChild(tick);
  }
  const entries = [];
  for (const [kind, arm] of all) {
    const points = arm.sweep.map(([t, , , f1]) => [(t - low) / span, f1]);
    drawCurve(svg, inner, points, kind, { label: SERIES[kind].label });

    // The threshold actually in force, chosen on training folds. Without this
    // the chart shows every cutoff except the one that produced the numbers.
    if (arm.threshold === undefined || arm.threshold === null) {
      entries.push({ kind, label: SERIES[kind].label, value: null });
      continue;
    }
    const x = inner.x0 + ((arm.threshold - low) / span) * inner.w;
    svg.appendChild(make("line", { class: "operating", stroke: SERIES[kind].color,
                                   x1: x, y1: inner.y0, x2: x, y2: PLOT.top }));
    svg.appendChild(make("circle", { class: "operating-dot", cx: x, cy: PLOT.top + 4,
                                     r: 3, fill: SERIES[kind].color }));
    // The legend reports F1 where that line stands, not the peak of the curve.
    // The peak is a number chosen with the answers in hand and no fit reaches it.
    const here = arm.sweep.reduce((best, row) =>
      Math.abs(row[0] - arm.threshold) < Math.abs(best[0] - arm.threshold) ? row : best,
      arm.sweep[0]);
    entries.push({ kind, label: SERIES[kind].label, value: fixed(here[3], 3) });
  }
  plotLegend(svg, inner, entries, "bl", "F1 in use");
  return makeInteractive(svg, inner, {
    xLabel: "threshold", yLabel: "F1",
    xOf: (fraction) => low + fraction * span,
  });
}

/* Five dots on a shared scale, one per held-out fold. The point of the figure
   is whether the arms' spreads overlap, and that only reads if every lane is
   drawn against the same axis with the axis actually shown. Each dot carries
   its fold and its score, so a reader can name the outlier rather than
   pointing at it. */
function foldStrip(arms) {
  const element = document.createElement("div");
  element.className = "folds";
  const present = armsOf(arms);
  const all = present.flatMap(([, arm]) => arm.per_fold_macro_f1);
  const low = Math.min(...all) - 0.02, high = Math.max(...all) + 0.02;
  const place = (value) => ((value - low) / (high - low)) * 100;
  const mid = (low + high) / 2;

  const lane = ([kind, arm]) => {
    const scores = arm.per_fold_macro_f1;
    const mean = scores.reduce((total, value) => total + value, 0) / (scores.length || 1);
    const dots = scores.map((score, index) =>
      `<i class="fold-dot" style="left:${place(score).toFixed(1)}%;` +
      `background:${SERIES[kind].color}" title="fold ${index + 1}: ${score.toFixed(4)}"></i>`).join("");
    // The mean as a rule through the dots, so two lanes can be compared without
    // reading the numbers at the end of each.
    const rule = `<i class="fold-mean" style="left:${place(mean).toFixed(1)}%;` +
      `background:${SERIES[kind].color}"></i>`;
    return `<div class="fold-lane"><span>${SERIES[kind].label}</span>
        <span class="fold-track">${rule}${dots}</span><b>${mean.toFixed(3)}</b></div>`;
  };

  element.innerHTML = `<div class="sc-title">Macro F1 by fold</div>
    <p class="sc-sub">Each dot is one held-out fold and the bar is their mean. Lanes that
      do not overlap mean the gap is consistent rather than an accident of one split.</p>
    ${present.map(lane).join("")}
    <div class="fold-axis"><span></span>
      <span class="fold-scale">
        <b style="left:0%">${low.toFixed(2)}</b>
        <b style="left:50%">${mid.toFixed(2)}</b>
        <b style="left:100%">${high.toFixed(2)}</b>
      </span><span>macro F1</span></div>`;
  return element;
}

/* ── the column editor ────────────────────────────────────────────────

   The comparison is only as good as the column set behind it, and the screen
   does not get the last word on that set. Every column can be toggled here,
   flagged or not, and measured again. The set measured last is tracked
   separately from the set currently selected, so the button can say honestly
   whether the numbers above still describe what is on screen.                */

let measuredSet = null;   // the drop list the visible results were computed from

function currentDrops() {
  return Object.entries(app.decisions)
    .filter(([, decision]) => decision === "drop")
    .map(([column]) => column)
    .sort();
}

function renderEditor() {
  if (!app.audit) return;
  const columns = app.audit.schema.feature_columns;
  $("editor").innerHTML = columns.map((column) => {
    const semantic = app.audit.semantic?.[column] || {};
    const flagged = semantic.verdict === "LEAK";
    const dropped = app.decisions[column] === "drop";
    return `<label class="ed-chip${dropped ? " dropped" : ""}${flagged ? " flagged" : ""}">
        <input type="checkbox" data-column="${escapeHtml(column)}"${dropped ? " checked" : ""}>
        <span class="ed-name">${escapeHtml(column)}</span>
        <span class="ed-tag">${flagged ? escapeHtml(semantic.mechanism || "LEAK") : "cleared"}</span>
      </label>`;
  }).join("");
  updateEditorFoot();
}

$("editor").addEventListener("change", (event) => {
  const box = event.target.closest("input[type=checkbox]");
  if (!box) return;
  app.decisions[box.dataset.column] = box.checked ? "drop" : "keep";
  box.closest(".ed-chip").classList.toggle("dropped", box.checked);
  updateEditorFoot();
  renderResultTable();
});

$("editor-preset").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  app.audit.schema.feature_columns.forEach((column) => {
    const flagged = app.audit.semantic?.[column]?.verdict === "LEAK";
    app.decisions[column] = button.dataset.preset === "flagged" && flagged ? "drop" : "keep";
  });
  renderEditor();
  renderResultTable();
});

function updateEditorFoot() {
  const drops = currentDrops();
  const measured = measuredSet !== null;
  const changed = !measured || drops.join("|") !== measuredSet.join("|");
  $("remeasure-btn").disabled = !changed || drops.length === 0;
  $("remeasure-btn").textContent = measured ? "Measure again" : "Measure this set";
  $("editor-note").textContent = drops.length === 0
    ? "With nothing dropped there are no two arms to compare."
    : !measured
      ? `${drops.length} column${drops.length === 1 ? "" : "s"} selected. Nothing has been `
        + `measured yet; the file below is already cleaned either way.`
      : changed
        ? `${drops.length} column${drops.length === 1 ? "" : "s"} selected — the results above were `
          + `measured on a different set.`
        : "The results above were measured on exactly this set.";
}

$("remeasure-btn").addEventListener("click", async () => {
  $("remeasure-btn").disabled = true;
  $("editor-note").textContent = "Refitting every arm on the new column set…";
  status("measuring", "busy");
  await runImpact({ keepEditor: true });
});

/* ── the dataset, before and after ── */

let csvView = "before";
$("csv-toggle").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  csvView = button.dataset.view;
  $("csv-toggle").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === button));
  renderResultTable();
});

function renderResultTable() {
  if (!app.table) return;
  const drops = new Set(Object.entries(app.decisions)
    .filter(([, decision]) => decision === "drop").map(([column]) => column));
  const columns = csvView === "before"
    ? app.table.columns
    : app.table.columns.filter((name) => !drops.has(name));

  $("csv-note").textContent = csvView === "before"
    ? `${app.table.columns.length} columns as uploaded.`
    : `${columns.length} columns, with ${drops.size} removed: ${[...drops].join(", ")}.`;

  const header = ["<th class=\"rownum\">#</th>"].concat(columns.map((name) =>
    `<th class="${drops.has(name) ? "dropped" : ""}">${escapeHtml(name)}</th>`)).join("");
  const body = app.table.rows.slice(0, 60).map((row, index) => {
    const cells = columns.map((name) =>
      `<td class="${drops.has(name) ? "dropped" : ""}">${escapeHtml(row[name] ?? "")}</td>`).join("");
    return `<tr><td class="rownum">${index + 1}</td>${cells}</tr>`;
  }).join("");
  $("result-table").innerHTML =
    `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;

  if (app.jobId) {
    $("report-link").href = `api/audit/${app.jobId}/report`;
    $("report-link").classList.remove("hidden");
  } else {
    $("report-link").classList.add("hidden");
  }
  // The note describes what the button will save, which is always the cleaned
  // set — not whichever view happens to be on screen.
  const kept = app.table.columns.length - drops.size;
  $("download-note").textContent =
    `${kept} columns × ${app.table.rows.length.toLocaleString()} rows` +
    (drops.size ? `, without ${[...drops].join(", ")}` : ", nothing removed");
}

/* The cleaned file is written here rather than fetched, so that the bytes the
   user saves are exactly the columns shown above — including any change made
   in the editor — and so the download works in the demo, which has no job. */

function toCsv(columns, rows) {
  const cell = (value) => {
    const text = value ?? "";
    return /[",\n]/.test(text) ? `"${String(text).replace(/"/g, '""')}"` : text;
  };
  return [columns.map(cell).join(",")]
    .concat(rows.map((row) => columns.map((name) => cell(row[name])).join(",")))
    .join("\n");
}

$("download-csv").addEventListener("click", () => {
  const drops = new Set(currentDrops());
  const columns = app.table.columns.filter((name) => !drops.has(name));
  const blob = new Blob([toCsv(columns, app.table.rows)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "cleaned.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  $("download-note").textContent =
    `Saved ${columns.length} columns × ${app.table.rows.length.toLocaleString()} rows` +
    (drops.size ? `, without ${[...drops].join(", ")}.` : ".");
});

/* ══════════ DEMO ══════════ */

const DEMO_COLUMNS = ["pclass", "name", "sex", "age", "sibsp", "parch", "ticket",
  "fare", "cabin", "embarked", "boat", "body", "home.dest"];

function demoAudit() {
  const clean = (reasons = []) =>
    ({ verdict: "OK", leak_votes: 0, shuffles_counted: 3, reasons });
  return {
    target: "survived",
    schema: { feature_columns: DEMO_COLUMNS },
    dictionary: null,
    semantic: {
      boat: { verdict: "LEAK", mechanism: "CONSEQUENCE", leak_votes: 3, shuffles_counted: 3,
        reasons: ["lifeboat identifier, recorded only for passengers who were rescued",
                  "a value here means the passenger survived; absence means they did not",
                  "cannot be known at boarding"] },
      body: { verdict: "LEAK", mechanism: "CONSEQUENCE", leak_votes: 3, shuffles_counted: 3,
        reasons: ["body recovery number, which exists only for passengers who died",
                  "assigned after the outcome, during recovery operations",
                  "the presence of any value is itself the label"] },
      cabin: clean(["cabin assignment is known at boarding",
                    "flagged once as possibly recorded after the fact, cleared by the other two orders"]),
      sex: clean(["demographic recorded at boarding; strongly predictive but legitimate"]),
      ...Object.fromEntries(DEMO_COLUMNS
        .filter((column) => !["boat", "body", "cabin", "sex"].includes(column))
        .map((column) => [column, clean()])),
    },
    statistical: {
      pclass: { correlation: -0.3125, flagged: false },
      name: { correlation: -0.2938, flagged: false },
      sex: { correlation: -0.5287, flagged: true },
      age: { correlation: -0.0555, flagged: false },
      sibsp: { correlation: -0.0278, flagged: false },
      parch: { correlation: 0.0827, flagged: false },
      ticket: { correlation: -0.2909, flagged: false },
      fare: { correlation: 0.2443, flagged: false },
      cabin: { correlation: -0.0948, flagged: false },
      embarked: { correlation: 0.0998, flagged: false },
      boat: { correlation: -0.0127, flagged: false },
      body: { correlation: null, flagged: false },
      "home.dest": { correlation: -0.2176, flagged: false },
    },
    buckets: {
      ...Object.fromEntries(DEMO_COLUMNS.map((column) => [column, "D"])),
      boat: "B", body: "B", sex: "C",
    },
  };
}

async function runDemo() {
  app.demo = true;
  // Say plainly that the screen is not running. The measurement further on is
  // real, and claiming the screen is too would put that in doubt as well.
  $("replay-note").classList.remove("hidden");
  $("demo-btn").disabled = true;
  $("demo-btn").textContent = "Loading the replay…";

  const text = await (await fetch("titanic.csv")).text();
  app.table = parseCsv(text);
  app.target = "survived";
  app.predictionPoint = "at boarding, before the voyage ends and before any rescue or recovery";
  app.model = "Qwen/Qwen3-Coder-480B-A35B-Instruct";

  $("topbar-file").textContent =
    `titanic.csv · ${app.table.columns.length} col · ${app.table.rows.length.toLocaleString()} rows`;
  renderData();
  $("target").value = "survived";
  $("prediction-point").value = app.predictionPoint;
  chooseTarget("survived");
  validateJob();

  ["data", "model", "detect", "review"].forEach(unlock);
  showTab("detect");
  status("running", "busy");

  for (let index = 0; index < STAGES.length; index++) {
    paintStages(index);
    $("run-state").textContent = `${STAGES[index][1]} — replayed`;
    setProgress(0.1 + index * 0.22);
    await sleep(620);
  }
  paintStages(STAGES.length);
  setProgress(1);
  $("run-state").textContent = "Replay complete — recorded verdicts loaded.";

  app.audit = demoAudit();
  status("audit complete", "go");
  markAudited();
  await settled();
  showFindings();
  $("demo-btn").disabled = false;
  $("demo-btn").textContent = "Replay the Titanic example";
}

/* ── returning visitors ────────────────────────────────────────────────
   The overview is a landing page, and a landing page is exactly what somebody
   on their second table does not need. Once an audit has finished here, the
   argument folds away behind a link and the upload moves to the top; the
   citation and the case for the tool stay one click away rather than four
   screens of scroll. First visit is unchanged. */

const SEEN_KEY = "crucible.hasRunAnAudit";

function markAudited() {
  try { localStorage.setItem(SEEN_KEY, "1"); } catch (error) { /* private mode */ }
}

function hasAuditedBefore() {
  try { return localStorage.getItem(SEEN_KEY) === "1"; } catch (error) { return false; }
}

function foldTheArgument() {
  const why = $("why");
  if (!why || why.dataset.folded) return;
  why.dataset.folded = "1";
  why.hidden = true;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "why-toggle";
  toggle.textContent = "Why this matters, and the paper behind it \u2192";
  toggle.addEventListener("click", () => {
    why.hidden = !why.hidden;
    toggle.textContent = why.hidden
      ? "Why this matters, and the paper behind it \u2192"
      : "Hide the argument \u2191";
    if (!why.hidden) why.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  why.parentNode.insertBefore(toggle, why);
}

/* ── start ── */
// Fetched at boot rather than on first use, so the Settings tab is correct
// whenever it is opened and the run button knows from the outset whether
// there is any quota to run on.
loadModels();
showTab("overview");
if (hasAuditedBefore()) foldTheArgument();
if (new URLSearchParams(location.search).has("demo")) runDemo();
