// Two scroll-driven walkthroughs: watermarking (five steps) and detection (three steps).
// Steps on the left pick the view; the stage on the right redraws for that view.
const $ = (id) => document.getElementById(id);
const IMG = "assets/lab/";
const V = { w: 500, h: 400, cx: 250, cy: 200, r: 176 };
const CODE_COLORS = ["#2f6fdf", "#e0702b", "#14937a", "#c23d8a", "#7a5bd6", "#a8820f"];
const MARGIN = 0.045;
const DETECT_P = 0.3;

const rand32 = () => crypto.getRandomValues(new Uint32Array(1))[0];
const state = {
  prompt: labPrompts[0], model: "sd21", planes: 2, angles: [],
  key: rand32(), seed: rand32(), check: "marked",
};

// ---------- Method ----------
function uniform(seed, salt) {
  let bits = (seed ^ salt) >>> 0;
  bits = Math.imul(bits ^ (bits >>> 16), 0x21f0aaad);
  bits = Math.imul(bits ^ (bits >>> 15), 0x735a2d97);
  return ((bits ^ (bits >>> 15)) >>> 0) / 4294967296 + 0.5 / 4294967296;
}
const scoreOf = (code, key = state.key) => uniform(key, 0x202a + parseInt(code, 2));
const imagesOf = () => state.prompt.images[state.model].map((image, index) => ({ ...image, index }));
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

function race(groups, key = state.key, seed = state.seed) {
  const rows = groups.map((g) => ({ ...g, score: scoreOf(g.code, key), value: Math.log(scoreOf(g.code, key)) / g.members.length }));
  const best = Math.max(...rows.map((r) => r.value));
  const tied = rows.filter((r) => r.value === best);
  const winner = tied[Math.floor(uniform(seed, 0x54100 + state.prompt.id) * tied.length)];
  const image = winner.members[Math.floor(uniform(seed, 0x7ae00 + state.prompt.id) * winner.members.length)];
  return { rows, winner, image };
}

function detect(image, key = state.key) {
  const code = codeOf(image);
  const codes = [code];
  marginsOf(image).forEach((m, i) => {
    if (Math.abs(m) < MARGIN) codes.push(code.slice(0, i) + (code[i] === "0" ? "1" : "0") + code.slice(i + 1));
  });
  const score = Math.max(...codes.map((c) => scoreOf(c, key)));
  // Without the watermark each region's score is uniform, so P(max of k scores > s) = 1 - s^k.
  // Threshold: a plain image passes with probability DETECT_P.
  const threshold = (1 - DETECT_P) ** (1 / codes.length);
  return { code, codes, score, threshold, p: 1 - score ** codes.length };
}

function drawPlanes() {
  const seeds = crypto.getRandomValues(new Uint32Array(state.planes + 2));
  const turn = (seeds[0] / 4294967296) * Math.PI;
  state.angles = Array.from({ length: state.planes }, (_, i) => {
    const jitter = (seeds[i + 2] / 4294967296 - 0.5) * 0.55;
    return turn + ((i + jitter) * Math.PI) / state.planes + ((seeds[1] >> i) & 1) * Math.PI;
  });
}

// ---------- Drawing helpers ----------
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
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const modelSelect = $("model");
const modelName = () => modelSelect.selectedOptions[0].textContent;
const colorMap = (groups) => new Map(groups.map((g, i) => [g.code, CODE_COLORS[i % CODE_COLORS.length]]));

function makeStage(root, tileCount) {
  const q = (name) => root.querySelector(`[data-${name}]`);
  const stage = {
    board: q("board"), regions: q("regions"), lines: q("lines"), dots: q("dots"),
    overlay: q("overlay"), caption: q("caption"), tiles: [],
  };
  for (let i = 0; i < tileCount; i += 1) {
    const fig = el("figure", "tile");
    const img = el("img");
    img.width = 176;
    img.height = 176;
    img.alt = "";
    fig.append(img, el("figcaption"));
    q("tiles").append(fig);
    stage.tiles.push(fig);
  }
  return stage;
}

function clear(stage) {
  stage.lines.replaceChildren();
  stage.dots.replaceChildren();
  stage.overlay.replaceChildren();
  stage.regions.style.opacity = 0;
}

function setTile(tile, { image, x, y, s, o = 1, label = "", color = "", winner = false }) {
  if (image) {
    const img = tile.querySelector("img");
    if (!img.src.endsWith(IMG + image.file)) img.src = IMG + image.file;
    img.alt = `${state.prompt.text}, candidate ${image.index + 1}`;
  }
  tile.style.left = pctX(x - s / 2);
  tile.style.top = pctY(y - s / 2);
  tile.style.width = pctX(s);
  tile.style.opacity = o;
  tile.classList.toggle("winner", winner);
  const cap = tile.querySelector("figcaption");
  cap.textContent = label;
  cap.style.background = color;
}

function svg(layer, tag, attrs) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  layer.append(n);
  return n;
}

function drawCircle(stage, withLines) {
  svg(stage.lines, "circle", { cx: V.cx, cy: V.cy, r: V.r, fill: withLines ? "none" : "#fff", stroke: "#c9d0db", "stroke-width": 1 });
  if (!withLines) return;
  for (const angle of state.angles) {
    const dx = V.r * -Math.sin(angle);
    const dy = -V.r * Math.cos(angle);
    svg(stage.lines, "line", { x1: V.cx - dx, y1: V.cy - dy, x2: V.cx + dx, y2: V.cy + dy, stroke: "#14161b", "stroke-width": 1.2, "stroke-dasharray": "5 4" });
  }
}

function paintRegions(stage, colors) {
  const ctx = stage.regions.getContext("2d");
  const scale = stage.regions.width / V.w;
  const cell = 2;
  ctx.clearRect(0, 0, stage.regions.width, stage.regions.height);
  for (let y = V.cy - V.r; y < V.cy + V.r; y += cell) {
    for (let x = V.cx - V.r; x < V.cx + V.r; x += cell) {
      if ((x + cell / 2 - V.cx) ** 2 + (y + cell / 2 - V.cy) ** 2 > V.r ** 2) continue;
      const code = codeOf({ x: (x + cell / 2 - V.cx) / (V.r * 0.92), y: -(y + cell / 2 - V.cy) / (V.r * 0.92) });
      const color = colors.get(code);
      ctx.fillStyle = color ? `${color}2e` : "#ffffff";
      ctx.fillRect(Math.round(x * scale), Math.round(y * scale), cell * scale, cell * scale);
    }
  }
  stage.regions.style.opacity = 1;
}

const dot = (stage, p, fill) => svg(stage.dots, "circle", { cx: p.x, cy: p.y, r: 5.5, fill, stroke: "#fff", "stroke-width": 2 });
const leader = (stage, p) => {
  if (p.x !== p.anchor.x || p.y !== p.anchor.y) {
    svg(stage.lines, "line", { x1: p.anchor.x, y1: p.anchor.y, x2: p.x, y2: p.y, stroke: "#14161b", "stroke-width": 1.2 });
  }
};

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

function ov(stage, cls, x, y, w) {
  const n = el("div", `ov ${cls}`);
  n.style.left = pctX(x);
  n.style.top = pctY(y);
  if (w !== undefined) n.style.width = pctX(w);
  stage.overlay.append(n);
  return n;
}

function scoreBar(stage, x, y, w, score, color) {
  const bar = ov(stage, "ov-bar", x, y, w);
  const fill = el("span");
  fill.style.width = `${score * 100}%`;
  fill.style.background = color;
  bar.append(fill);
  return bar;
}

// ---------- Watermarking ----------
const gen = makeStage($("gen-stage"), 6);

const genViews = [
  function samples(stage, { images }) {
    const s = 128;
    images.forEach((image, i) => setTile(stage.tiles[i], {
      image, s, x: V.cx + ((i % 3) - 1) * (s + 16), y: V.cy + (i < 3 ? -1 : 1) * (s / 2 + 8),
    }));
    stage.caption.innerHTML = `N = 6 samples of <strong>“${escapeHtml(state.prompt.text)}”</strong> from ${modelName()}.`;
  },
  function embed(stage, { images }) {
    drawCircle(stage, false);
    placeOnPlot(images, 54).forEach((p, i) => {
      leader(stage, p);
      dot(stage, p.anchor, "#14161b");
      setTile(stage.tiles[i], { image: images[i], x: p.x, y: p.y, s: 54 });
    });
    stage.caption.textContent = "Each dot is one image's embedding. Similar images land close together.";
  },
  function regions(stage, { images, groups, colors }) {
    paintRegions(stage, colors);
    drawCircle(stage, true);
    placeOnPlot(images, 46).forEach((p, i) => {
      const code = codeOf(images[i]);
      leader(stage, p);
      dot(stage, p.anchor, colors.get(code));
      setTile(stage.tiles[i], { image: images[i], x: p.x, y: p.y, s: 46, label: code, color: colors.get(code) });
    });
    stage.caption.innerHTML = `M = ${state.planes} hyperplanes give each image a ${state.planes}-bit code. These six images fall into <strong>${groups.length}</strong> regions.`;
  },
  function scores(stage, { groups, colors }) {
    const n = groups.length;
    const rowH = Math.min(78, 340 / n);
    const y = (i) => V.cy - (n * rowH) / 2 + rowH / 2 + i * rowH;
    const most = Math.max(...groups.map((g) => g.members.length));
    const size = Math.min(60, rowH - 12, 186 / most - 6);
    groups.forEach((g, i) => {
      const row = ov(stage, "ov-row", 28, y(i));
      const chip = el("span", "chip");
      chip.style.background = colors.get(g.code);
      const text = el("span");
      text.append(el("span", "ov-code", g.code), el("br"), el("span", "ov-muted", `${g.members.length} ${g.members.length === 1 ? "image" : "images"}`));
      row.append(chip, text);
      g.members.forEach((image, j) => setTile(stage.tiles[image.index], { image, x: 140 + size / 2 + j * (size + 6), y: y(i), s: size }));
      scoreBar(stage, 338, y(i), 100, scoreOf(g.code), colors.get(g.code));
      ov(stage, "ov-score", 446, y(i)).textContent = scoreOf(g.code).toFixed(2);
    });
    ov(stage, "ov-muted", 338, y(0) - 38).textContent = "Key score";
    stage.caption.innerHTML = `Key <code>${hex(state.key)}</code> gives every region a score between 0 and 1.`;
  },
  function pick(stage, { images, colors, pick: chosen }) {
    const others = images.filter((im) => im !== chosen.image);
    setTile(stage.tiles[chosen.image.index], { image: chosen.image, x: V.cx, y: 158, s: 196, winner: true });
    others.forEach((im, j) => setTile(stage.tiles[im.index], { image: im, x: V.cx + (j - 2) * 54, y: 356, s: 44, o: 0.35 }));
    const label = ov(stage, "ov-label", V.cx, 266);
    const chip = el("span", "chip");
    chip.style.background = colors.get(chosen.winner.code);
    label.append(chip, " ", el("strong", "", "Returned image"), ` · region ${chosen.winner.code}`);
    const n = chosen.winner.members.length;
    stage.caption.innerHTML = `Region <strong>${chosen.winner.code}</strong> wins: score ${chosen.winner.score.toFixed(2)}, ${n} ${n === 1 ? "candidate" : "candidates"}. ` +
      `The winner maximises <code>log(score) / candidates</code>, which keeps the output distribution unchanged. Pixels are untouched.`;
  },
];

// ---------- Detection ----------
const det = makeStage($("det-stage"), 1);

// A worked example: the first fixed key whose watermarked image clears the threshold
// while another candidate (a plain sample) stays below it.
function detectionExample(images, groups) {
  let fallback;
  for (let i = 1; i <= 400; i += 1) {
    const key = Math.imul(i, 0x9e3779b9) >>> 0;
    const seed = Math.imul(i, 0x85ebca6b) >>> 0;
    const marked = race(groups, key, seed).image;
    const markedResult = detect(marked, key);
    // Prefer another candidate of this prompt; otherwise any image generated for another prompt.
    const others = labPrompts.filter((p) => p !== state.prompt).flatMap((p) => p.images[state.model].map((im, index) => ({ ...im, index })));
    const plain = [...images.filter((im) => im !== marked), ...others].find((im) => {
      const r = detect(im, key);
      return r.score <= r.threshold;
    });
    fallback ??= { key, marked, plain: plain ?? images.find((im) => im !== marked) ?? marked };
    if (markedResult.score > markedResult.threshold && plain) return { key, marked, plain };
  }
  return fallback;
}

const detViews = [
  function take(stage, { image }) {
    setTile(stage.tiles[0], { image, x: V.cx, y: V.cy, s: 250 });
    stage.caption.textContent = "";
  },
  function locate(stage, { image, result }) {
    const colors = new Map(result.codes.map((c, i) => [c, i === 0 ? "#14161b" : "#9aa3b2"]));
    paintRegions(stage, colors);
    drawCircle(stage, true);
    const [p] = placeOnPlot([image], 90);
    leader(stage, p);
    dot(stage, p.anchor, "#14161b");
    setTile(stage.tiles[0], { image, x: p.x, y: p.y, s: 90, label: result.code, color: "#14161b" });
    stage.caption.innerHTML = `The image lands in region <strong>${result.code}</strong>.` +
      (result.codes.length > 1 ? ` It sits close to a hyperplane, so the neighbouring region ${result.codes.slice(1).join(", ")} is checked too, in case an edit moved it.` : "");
  },
  function score(stage, { image, result }) {
    setTile(stage.tiles[0], { image, x: 110, y: 150, s: 150 });
    const detected = result.score > result.threshold;
    ov(stage, "ov-muted", 220, 92).textContent = `Key score of region ${result.codes.join(" / ")}`;
    scoreBar(stage, 220, 120, 210, result.score, "#14161b");
    const tick = ov(stage, "ov-thresh", 220 + 210 * result.threshold, 120);
    tick.append(el("span", "", `threshold ${result.threshold.toFixed(2)}`));
    ov(stage, "ov-score", 440, 120).textContent = result.score.toFixed(2);
    const verdict = ov(stage, "ov-verdict", 220, 176);
    verdict.append(el("strong", "", detected ? "Watermark detected" : "No watermark detected"),
      el("span", "", `Score ${result.score.toFixed(2)} is ${detected ? "above" : "below"} the threshold.`));
    stage.caption.textContent = "";
  },
];

// ---------- Rendering and scroll sync ----------
const sections = [
  { root: document.querySelector('[data-section="gen"]'), stage: gen, views: genViews, step: 0 },
  { root: document.querySelector('[data-section="det"]'), stage: det, views: detViews, step: 0 },
];

function render() {
  const images = imagesOf();
  const groups = groupsOf(images);
  const chosen = race(groups);
  const example = detectionExample(images, groups);
  const image = state.check === "marked" ? example.marked : example.plain;
  const ctx = { images, groups, colors: colorMap(groups), pick: chosen, image, result: detect(image, example.key) };
  document.querySelectorAll("[data-key-value]").forEach((n) => (n.textContent = `key ${hex(state.key)}`));
  for (const section of sections) {
    clear(section.stage);
    section.views[section.step](section.stage, ctx);
  }
}

const narrow = matchMedia("(max-width: 980px)");
function syncSteps() {
  const line = innerHeight * (narrow.matches ? 0.72 : 0.5);
  let changed = false;
  for (const section of sections) {
    let best = 0;
    let bestDistance = Infinity;
    for (const s of section.root.querySelectorAll(".step")) {
      const r = s.getBoundingClientRect();
      const distance = r.top > line ? r.top - line : r.bottom < line ? line - r.bottom : 0;
      if (distance < bestDistance) { bestDistance = distance; best = Number(s.dataset.step); }
    }
    if (best !== section.step) changed = true;
    section.step = best;
    section.root.querySelectorAll(".step").forEach((s) => s.classList.toggle("is-active", Number(s.dataset.step) === best));
  }
  if (changed) render();
}
addEventListener("scroll", syncSteps, { passive: true });
addEventListener("resize", syncSteps);

// ---------- Controls ----------
function updatePrompt() {
  $("prompt-name").textContent = state.prompt.text;
  $("prompt-preview").src = IMG + state.prompt.images[state.model][0].file;
  render();
}

modelSelect.addEventListener("change", () => {
  state.model = modelSelect.value;
  updatePrompt();
});

function pressOnly(selector, button) {
  document.querySelectorAll(selector).forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
}

document.querySelectorAll("[data-planes]").forEach((button) => button.addEventListener("click", () => {
  state.planes = Number(button.dataset.planes);
  pressOnly("[data-planes]", button);
  drawPlanes();
  render();
}));

document.querySelectorAll("[data-check]").forEach((button) => button.addEventListener("click", () => {
  state.check = button.dataset.check;
  pressOnly("[data-check]", button);
  render();
}));

document.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => {
  const action = button.dataset.action;
  if (action === "planes") drawPlanes();
  if (action === "key") { state.key = rand32(); state.seed = rand32(); }
  render();
}));

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

drawPlanes();
render();
syncSteps();

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
