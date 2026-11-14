// Scroll-driven walkthrough. The steps on the left set state.step; the stage on the right
// re-lays out the same six candidate tiles for each step.
const $ = (id) => document.getElementById(id);
const IMG = "assets/lab/";
const V = { w: 500, h: 400, cx: 250, cy: 200, r: 176 };
const CODE_COLORS = ["#2f6fdf", "#e0702b", "#14937a", "#c23d8a", "#7a5bd6", "#a8820f"];
const MARGIN = 0.045;
const CUTOFF_QUANTILE = 0.9;
const reduced = matchMedia("(prefers-reduced-motion: reduce)");

const rand32 = () => crypto.getRandomValues(new Uint32Array(1))[0];
const state = {
  prompt: labPrompts[0], model: "sd21", planes: 2, angles: [],
  key: rand32(), seed: rand32(), cleanSeed: rand32(),
  step: 0, tally: [0, 0, 0, 0, 0, 0], runs: 0,
};
let tallyTimer = 0;

// ---------- Method (same rules as the PR's walkthrough) ----------
function uniform(seed, salt) {
  let bits = (seed ^ salt) >>> 0;
  bits = Math.imul(bits ^ (bits >>> 16), 0x21f0aaad);
  bits = Math.imul(bits ^ (bits >>> 15), 0x735a2d97);
  return ((bits ^ (bits >>> 15)) >>> 0) / 4294967296 + 0.5 / 4294967296;
}
const scoreOf = (key, code) => uniform(key, 0x202a + parseInt(code, 2));
const imagesOf = (prompt = state.prompt) => prompt.images[state.model].map((image, index) => ({ ...image, index }));
const marginsOf = (image) => state.angles.map((a) => image.x * Math.cos(a) + image.y * Math.sin(a));
const codeOf = (image) => marginsOf(image).map((m) => (m >= 0 ? "1" : "0")).join("");

function groupsOf(images) {
  const groups = new Map();
  for (const image of images) {
    const code = codeOf(image);
    if (!groups.has(code)) groups.set(code, []);
    groups.get(code).push(image);
  }
  return [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([code, members]) => ({ code, members }));
}

function race(groups, key, seed, promptId) {
  const rows = groups.map((g) => {
    const score = scoreOf(key, g.code);
    return { ...g, score, value: Math.log(score) / g.members.length };
  });
  const best = Math.max(...rows.map((r) => r.value));
  const tied = rows.filter((r) => r.value === best);
  const winner = tied[Math.floor(uniform(seed, 0x54100 + promptId) * tied.length)];
  const image = winner.members[Math.floor(uniform(seed, 0x7ae00 + promptId) * winner.members.length)];
  return { rows, winner, image };
}

function detectScore(image, key) {
  const code = codeOf(image);
  const codes = [code];
  marginsOf(image).forEach((m, i) => {
    if (Math.abs(m) < MARGIN) codes.push(code.slice(0, i) + (code[i] === "0" ? "1" : "0") + code.slice(i + 1));
  });
  return Math.max(...codes.map((c) => scoreOf(key, c)));
}

function cutoff(prompts, key) {
  // Exact clean distribution of the batch average when each image is a uniform candidate.
  let sums = [0];
  for (const prompt of prompts) {
    const scores = imagesOf(prompt).map((image) => detectScore(image, key));
    sums = sums.flatMap((s) => scores.map((x) => s + x));
  }
  sums.sort((a, b) => a - b);
  return sums[Math.floor(CUTOFF_QUANTILE * sums.length)] / prompts.length;
}

function drawPlanes() {
  const seeds = crypto.getRandomValues(new Uint32Array(state.planes + 2));
  const turn = (seeds[0] / 4294967296) * Math.PI;
  state.angles = Array.from({ length: state.planes }, (_, i) => {
    const jitter = (seeds[i + 2] / 4294967296 - 0.5) * 0.55;
    return turn + ((i + jitter) * Math.PI) / state.planes + ((seeds[1] >> i) & 1) * Math.PI;
  });
}

// ---------- Helpers ----------
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const pctX = (x) => `${(x / V.w) * 100}%`;
const pctY = (y) => `${(y / V.h) * 100}%`;
const hex = (key) => `0x${key.toString(16).padStart(8, "0")}`;
const toBoard = (image) => ({ x: V.cx + V.r * image.x * 0.92, y: V.cy - V.r * image.y * 0.92 });

function colorMap(groups) {
  return new Map(groups.map((g, i) => [g.code, CODE_COLORS[i % CODE_COLORS.length]]));
}

function placeOnPlot(images, size) {
  // Nearest free spot, searched in rings around the true point. Leave room for the code label.
  const wide = size + 8;
  const tall = size + 22;
  const slots = [[0, 0]];
  for (let ring = 1; ring <= 4; ring += 1) {
    for (let k = 0; k < 12; k += 1) {
      const t = (k / 12) * 2 * Math.PI;
      slots.push([Math.cos(t) * ring * wide * 0.7, Math.sin(t) * ring * tall * 0.7]);
    }
  }
  const placed = [];
  return images.map((image) => {
    const p = toBoard(image);
    const options = slots.map(([dx, dy]) => ({
      x: Math.min(V.w - size / 2 - 6, Math.max(size / 2 + 6, p.x + dx)),
      y: Math.min(V.h - size / 2 - 22, Math.max(size / 2 + 6, p.y + dy)),
    }));
    const spot = options.find((o) => placed.every((q) => Math.abs(o.x - q.x) >= wide || Math.abs(o.y - q.y) >= tall)) ?? options[0];
    placed.push(spot);
    return { ...spot, anchor: p };
  });
}

// ---------- Stage ----------
const board = $("board");
const tilesLayer = $("tiles");
const overlay = $("overlay");
const lines = $("lines");
const regions = $("regions");
const caption = $("stage-caption");
const tiles = Array.from({ length: 6 }, () => {
  const fig = el("figure", "tile");
  const img = el("img");
  img.width = 176;
  img.height = 176;
  img.alt = "";
  fig.append(img, el("figcaption"));
  tilesLayer.append(fig);
  return fig;
});

function setTile(tile, { x, y, s, o = 1, label = "", color = "", winner = false }) {
  tile.style.left = pctX(x - s / 2);
  tile.style.top = pctY(y - s / 2);
  tile.style.width = pctX(s);
  tile.style.opacity = o;
  tile.classList.toggle("winner", winner);
  const cap = tile.querySelector("figcaption");
  cap.textContent = label;
  cap.style.background = color;
}

function svg(tag, attrs) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  lines.append(n);
  return n;
}

function drawCircle(withLines) {
  svg("circle", { cx: V.cx, cy: V.cy, r: V.r, fill: withLines ? "none" : "#fff", stroke: "#c9d0db", "stroke-width": 1 });
  if (!withLines) return;
  for (const angle of state.angles) {
    // Points with x·cos + y·sin = 0 lie along direction (-sin, cos).
    const dx = V.r * -Math.sin(angle);
    const dy = -V.r * Math.cos(angle);
    svg("line", { x1: V.cx - dx, y1: V.cy - dy, x2: V.cx + dx, y2: V.cy + dy, stroke: "#14161b", "stroke-width": 1.2, "stroke-dasharray": "5 4" });
  }
}

function paintRegions(colors) {
  const ctx = regions.getContext("2d");
  const scale = regions.width / V.w;
  ctx.clearRect(0, 0, regions.width, regions.height);
  const cell = 2;
  for (let y = V.cy - V.r; y < V.cy + V.r; y += cell) {
    for (let x = V.cx - V.r; x < V.cx + V.r; x += cell) {
      const px = (x + cell / 2 - V.cx) / (V.r * 0.92);
      const py = -(y + cell / 2 - V.cy) / (V.r * 0.92);
      if ((x + cell / 2 - V.cx) ** 2 + (y + cell / 2 - V.cy) ** 2 > V.r ** 2) continue;
      const color = colors.get(codeOf({ x: px, y: py }));
      ctx.fillStyle = color ? `${color}2e` : "#ffffff";
      ctx.fillRect(Math.round(x * scale), Math.round(y * scale), cell * scale, cell * scale);
    }
  }
}

function ov(cls, x, y, w) {
  const n = el("div", `ov ${cls}`);
  n.style.left = pctX(x);
  n.style.top = pctY(y);
  if (w !== undefined) n.style.width = pctX(w);
  overlay.append(n);
  return n;
}

function render() {
  const images = imagesOf();
  const groups = groupsOf(images);
  const colors = colorMap(groups);
  const pick = race(groups, state.key, state.seed, state.prompt.id);
  const step = state.step;

  tiles.forEach((tile, i) => {
    const img = tile.querySelector("img");
    const src = IMG + images[i].file;
    if (!img.src.endsWith(src)) img.src = src;
    img.alt = `${state.prompt.text}, candidate ${i + 1}`;
  });
  lines.replaceChildren();
  overlay.replaceChildren();
  regions.style.opacity = step === 2 ? 1 : 0;
  board.classList.toggle("is-detect", step === 6);
  document.querySelectorAll("[data-key-value]").forEach((n) => (n.textContent = `key ${hex(state.key)}`));

  const views = [viewGrid, viewPlot, viewCodes, viewScores, viewPick, viewFair, viewDetect];
  views[step]({ images, groups, colors, pick });
}

function viewGrid({ images }) {
  const s = 128;
  images.forEach((_, i) => setTile(tiles[i], { x: V.cx + ((i % 3) - 1) * (s + 16), y: V.cy + (i < 3 ? -1 : 1) * (s / 2 + 8), s, label: `#${i + 1}`, color: "#14161b" }));
  caption.innerHTML = `Six samples of <strong>“${escapeHtml(state.prompt.text)}”</strong> from ${modelName()}.`;
}

function viewPlot({ images }) {
  drawCircle(false);
  const spots = placeOnPlot(images, 62);
  spots.forEach((p, i) => {
    if (p.x !== p.anchor.x || p.y !== p.anchor.y) {
      svg("line", { x1: p.anchor.x, y1: p.anchor.y, x2: p.x, y2: p.y, stroke: "#9aa3b2", "stroke-width": 1 });
    }
    svg("circle", { cx: p.anchor.x, cy: p.anchor.y, r: 3, fill: "#14161b" });
    setTile(tiles[i], { x: p.x, y: p.y, s: 62 });
  });
  caption.textContent = "Each dot is one image's embedding. Nearby dots are images the encoder sees as similar.";
}

function viewCodes({ images, groups, colors }) {
  paintRegions(colors);
  drawCircle(true);
  const spots = placeOnPlot(images, 56);
  spots.forEach((p, i) => {
    svg("circle", { cx: p.anchor.x, cy: p.anchor.y, r: 3, fill: "#14161b" });
    if (p.x !== p.anchor.x || p.y !== p.anchor.y) svg("line", { x1: p.anchor.x, y1: p.anchor.y, x2: p.x, y2: p.y, stroke: "#9aa3b2", "stroke-width": 1 });
    const code = codeOf(images[i]);
    setTile(tiles[i], { x: p.x, y: p.y, s: 56, label: code, color: colors.get(code) });
  });
  caption.innerHTML = `<strong>${state.planes} lines</strong> cut the plot into ${2 * state.planes} regions, each with its own code. These six images fall into <strong>${groups.length}</strong> of them.` +
    (groups.length === 1 ? " Redraw the lines to split them up." : "");
}

function rowLayout(n) {
  const rowH = Math.min(78, 340 / n);
  const top = V.cy - (n * rowH) / 2 + rowH / 2;
  return { rowH, y: (i) => top + i * rowH, size: Math.min(60, rowH - 12) };
}

function viewScores({ groups, colors }, highlight = null) {
  const { y, size: rowSize } = rowLayout(groups.length);
  const most = Math.max(...groups.map((g) => g.members.length));
  const size = Math.min(rowSize, 186 / most - 6);
  groups.forEach((g, i) => {
    const row = ov("ov-row", 28, y(i));
    const chip = el("span", "chip");
    chip.style.background = colors.get(g.code);
    const text = el("span");
    text.append(el("span", "ov-code", g.code), el("br"), el("span", "ov-muted", `${g.members.length} ${g.members.length === 1 ? "image" : "images"}`));
    row.append(chip, text);
    g.members.forEach((image, j) => setTile(tiles[image.index], { x: 140 + size / 2 + j * (size + 6), y: y(i), s: size }));
    const score = scoreOf(state.key, g.code);
    const bar = ov("ov-bar", 338, y(i), 100);
    const fill = el("span");
    fill.style.width = `${score * 100}%`;
    fill.style.background = colors.get(g.code);
    bar.append(fill);
    ov("ov-score", 446, y(i)).textContent = score.toFixed(2);
  });
  ov("ov-muted", 338, y(0) - 38).textContent = "Key score";
  const top = groups.reduce((a, b) => (scoreOf(state.key, b.code) > scoreOf(state.key, a.code) ? b : a));
  caption.innerHTML = `With key <code>${hex(state.key)}</code>, code <strong>${top.code}</strong> scores highest (${scoreOf(state.key, top.code).toFixed(2)}).`;
}

function viewPick({ images, pick, colors }) {
  const big = 196;
  const others = images.filter((im) => im !== pick.image);
  setTile(tiles[pick.image.index], { x: V.cx, y: 158, s: big, winner: true });
  others.forEach((im, j) => setTile(tiles[im.index], { x: V.cx + (j - 2) * 54, y: 356, s: 44, o: 0.35 }));
  const label = ov("ov-label", V.cx, 266);
  const chip = el("span", "chip");
  chip.style.background = colors.get(pick.winner.code);
  label.append(chip, " ", el("strong", "", `Returned: #${pick.image.index + 1}`), ` · code ${pick.winner.code}`);
  $("race-values").textContent = pick.rows.map((r) => `${r.code}: log(${r.score.toFixed(2)}) / ${r.members.length} = ${r.value.toFixed(2)}`).join("\n");
  $("race-values").style.whiteSpace = "pre-line";
  const size = pick.winner.members.length;
  caption.innerHTML = `Code <strong>${pick.winner.code}</strong> wins the race` +
    (size > 1 ? `, and one of its ${size} images is drawn at random.` : ".") +
    " The other candidates are discarded.";
}

function viewFair({ images }) {
  const s = 62;
  const base = 350;
  const maxH = 210;
  const total = Math.max(1, state.runs);
  images.forEach((_, i) => {
    const x = V.cx + (i - 2.5) * 74;
    setTile(tiles[i], { x, y: 62, s });
    const share = state.tally[i] / total;
    const bar = ov("ov-tally", x, base, 30);
    bar.style.top = "auto";
    bar.style.bottom = pctY(V.h - base);
    bar.style.height = pctY(Math.min(maxH, (share / 0.3) * maxH));
    ov("ov-pct", x, base + 8).textContent = state.runs ? `${Math.round(share * 100)}%` : "";
  });
  const expectY = base - (1 / 6 / 0.3) * maxH;
  const line = ov("ov-expect", 36, expectY, V.w - 72);
  line.append(el("span", "", "fair share, 1 in 6"));
  caption.innerHTML = state.runs
    ? `<strong>${state.runs.toLocaleString()} random keys.</strong> Each candidate was returned about one time in six.`
    : "Running random keys…";
}

function detectionPrompts() {
  const at = labPrompts.indexOf(state.prompt);
  return [state.prompt, ...[5, 10, 15].map((k) => labPrompts[(at + k) % labPrompts.length])];
}

function viewDetect() {
  tiles.forEach((tile) => setTile(tile, { x: V.cx, y: V.cy, s: 120, o: 0 }));
  const prompts = detectionPrompts();
  const marked = prompts.map((p) => race(groupsOf(imagesOf(p)), state.key, state.seed, p.id).image);
  const clean = prompts.map((p, i) => {
    const ims = imagesOf(p);
    return ims[Math.floor(uniform(state.cleanSeed, 0x3c1 + i * 977 + p.id) * ims.length)];
  });
  const cut = cutoff(prompts, state.key);
  const panel = el("div", "detect");
  const batch = (title, list, cls) => {
    const scores = list.map((im) => detectScore(im, state.key));
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const box = el("div", `batch ${cls}`);
    const thumbs = el("div", "batch-thumbs");
    list.forEach((im, i) => {
      const fig = el("figure");
      const img = el("img");
      img.src = IMG + im.file;
      img.alt = `${prompts[i].text}`;
      fig.append(img, el("figcaption", "", scores[i].toFixed(2)));
      thumbs.append(fig);
    });
    const avgBox = el("div", "batch-avg");
    const verdict = cls === "marked"
      ? (avg > cut ? "Watermark detected" : "Missed with this key")
      : (avg > cut ? "False alarm" : "Not flagged");
    avgBox.append(el("strong", "", avg.toFixed(2)), el("span", "", verdict));
    box.append(el("h4", "", title), thumbs, avgBox);
    return { box, avg };
  };
  const a = batch("Selected with your key", marked, "marked");
  const b = batch("Plain samples, no watermark", clean, "clean");
  const meter = el("div", "meter");
  meter.innerHTML = `<div class="meter-track"></div><div class="meter-ends"><span>0</span><span>1</span></div>`;
  const cutEl = el("div", "meter-cut");
  cutEl.style.left = `${cut * 100}%`;
  const dot = (avg, cls) => {
    const d = el("div", `meter-dot ${cls}`);
    d.style.left = `${avg * 100}%`;
    return d;
  };
  meter.append(cutEl, dot(b.avg, "clean"), dot(a.avg, "marked"));
  const legend = el("p", "meter-legend");
  legend.innerHTML = `<span><i class="meter-dot marked"></i>your batch</span><span><i class="meter-dot clean"></i>plain batch</span><span><i class="meter-cut"></i>cutoff ${cut.toFixed(2)}</span>`;
  const meterBox = el("div", "meter-box");
  meterBox.append(meter, legend);
  panel.append(a.box, b.box, meterBox);
  overlay.append(panel);
  caption.innerHTML = `A batch is flagged when its average beats the cutoff, set so only 10% of plain batches from these prompts pass. ` +
    `With ${state.planes} lines and 4 images the demo is noisy. The paper uses longer codes and larger batches.`;
}

// ---------- Fairness tally ----------
function runTally(count) {
  cancelAnimationFrame(tallyTimer);
  const images = imagesOf();
  const groups = groupsOf(images);
  let left = count;
  const tick = () => {
    const n = reduced.matches ? left : Math.min(left, 40);
    for (let i = 0; i < n; i += 1) {
      state.tally[race(groups, rand32(), rand32(), state.prompt.id).image.index] += 1;
    }
    state.runs += n;
    left -= n;
    if (state.step === 5) render();
    if (left > 0) tallyTimer = requestAnimationFrame(tick);
  };
  tick();
}
function resetTally() {
  cancelAnimationFrame(tallyTimer);
  state.tally = [0, 0, 0, 0, 0, 0];
  state.runs = 0;
}

// ---------- Controls ----------
const modelSelect = $("model");
const modelName = () => modelSelect.selectedOptions[0].textContent;
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function setStep(step) {
  if (step === state.step) return;
  state.step = step;
  document.querySelectorAll(".step").forEach((s) => s.classList.toggle("is-active", Number(s.dataset.step) === step));
  if (step === 5 && state.runs === 0) runTally(1000);
  render();
}

function updatePrompt() {
  $("prompt-name").textContent = state.prompt.text;
  $("prompt-preview").src = IMG + state.prompt.images[state.model][0].file;
  resetTally();
  if (state.step === 5) runTally(1000);
  render();
}

modelSelect.addEventListener("change", () => {
  state.model = modelSelect.value;
  updatePrompt();
});

document.querySelectorAll("[data-planes]").forEach((button) => {
  button.addEventListener("click", () => {
    state.planes = Number(button.dataset.planes);
    document.querySelectorAll("[data-planes]").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    drawPlanes();
    resetTally();
    render();
  });
});

document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.dataset.action;
    if (action === "planes") { drawPlanes(); resetTally(); }
    if (action === "key") { state.key = rand32(); state.seed = rand32(); }
    if (action === "clean") state.cleanSeed = rand32();
    if (action === "tally") { runTally(1000); return; }
    render();
  });
});

const dialog = $("prompt-dialog");
$("open-prompts").addEventListener("click", () => {
  const grid = $("prompt-grid");
  grid.replaceChildren(...labPrompts.map((prompt) => {
    const button = el("button", "prompt-option");
    button.type = "button";
    button.setAttribute("aria-pressed", String(prompt === state.prompt));
    const img = el("img");
    img.src = IMG + prompt.images[state.model][0].file;
    img.alt = "";
    img.loading = "lazy";
    button.append(img, el("span", "", prompt.text));
    button.addEventListener("click", () => {
      state.prompt = prompt;
      dialog.close();
      updatePrompt();
    });
    return button;
  }));
  dialog.showModal();
  grid.querySelector('[aria-pressed="true"]').focus();
});

// Activate the step closest to the trigger line. Recomputed on every scroll frame, so jumps
// (anchor links, fast scrolling) land on the right step.
const narrow = matchMedia("(max-width: 980px)");
const stepEls = [...document.querySelectorAll(".step")];
function syncStep() {
  const line = innerHeight * (narrow.matches ? 0.72 : 0.5);
  let best = 0;
  let bestDistance = Infinity;
  for (const s of stepEls) {
    const r = s.getBoundingClientRect();
    const distance = r.top > line ? r.top - line : r.bottom < line ? line - r.bottom : 0;
    if (distance < bestDistance) { bestDistance = distance; best = Number(s.dataset.step); }
  }
  setStep(best);
}
addEventListener("scroll", syncStep, { passive: true });
addEventListener("resize", syncStep);

drawPlanes();
document.querySelector('.step[data-step="0"]').classList.add("is-active");
render();
syncStep();

const copy = $("copy-citation");
const citation = $("citation-text");
copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copy.textContent = "Copied";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    copy.textContent = "Text selected";
  }
});
