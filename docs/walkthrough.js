const demo = document.querySelector("#walkthrough");
const byId = (id) => document.getElementById(id);
const previous = byId("previous");
const next = byId("next");
const status = byId("step-status");
const detectionNext = byId("detection-next");
const detectionPrevious = byId("detection-previous");
const model = byId("model");
const dialog = byId("prompt-dialog");
const promptGrid = byId("prompt-grid");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const imageRoot = "assets/lab/";
const generationStages = ["Generate candidates", "Group by image code", "Return one image"];
const detectionStages = ["Collect a batch", "Score each image", "Make a batch decision"];
const generationActions = ["Show hyperplanes", "Group images", "Select an image", "Replay"];
const detectionActions = ["Check image codes", "Apply the secret key", "Check the batch", "Replay"];
const stationForStep = [0, 1, 1, 2];
const stepNotes = [
  "Six outputs from the same prompt.",
  "",
  "Images with the same code go in the same group.",
  "Gumbel selection uses both the key score and the group size. Then it draws one image from the winning group.",
  "Four images selected with the same key, one per prompt.",
  "For images close to a line, also try flipping the corresponding bit.",
  "Keep the highest keyed score among the codes tried for each image.",
  "Average the scores and compare the result with the cutoff from clean batches.",
];
const marginLimit = 0.045;
const exampleQuantile = 0.8;
const imagesFor = (prompt) => prompt.images[model.value];
const randomSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
const state = { prompt: labPrompts[0], key: null, samplerSeed: randomSeed(), angles: null, step: 0, detectionStep: -1 };
const activeAnimations = [];
const movingTiles = [];

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function makeTile(image, text, className = "") {
  const tile = node("figure", `tile ${className}`);
  const img = node("img");
  img.src = imageRoot + image.file;
  img.alt = image.label;
  img.width = 192;
  img.height = 192;
  img.dataset.image = image.file;
  tile.append(img, node("figcaption", "", text));
  return tile;
}

function candidates(prompt) {
  return imagesFor(prompt).map((image, index) => ({
    ...image,
    label: `${prompt.text}, candidate ${index + 1}`,
    index,
  }));
}

function wrapImages(items, caption) {
  const grid = node("div", "image-grid");
  grid.append(...items.map((image, index) => makeTile(image, caption(image, index))));
  return grid;
}

// A small repeatable mixer serves as the demo's keyed score function.
function uniform(seed, salt) {
  let bits = (seed ^ salt) >>> 0;
  bits = Math.imul(bits ^ (bits >>> 16), 0x21f0aaad);
  bits = Math.imul(bits ^ (bits >>> 15), 0x735a2d97);
  return ((bits ^ (bits >>> 15)) >>> 0) / 4294967296 + 0.5 / 4294967296;
}

function keyedScore(code) {
  return uniform(state.key, 0x202a + parseInt(code, 2));
}

function margins(image) {
  return state.angles.map((angle) => image.x * Math.cos(angle) + image.y * Math.sin(angle));
}

function codeFor(image) {
  return margins(image).map((margin) => margin >= 0 ? "1" : "0").join("");
}

function groupImages(images) {
  const groups = new Map();
  for (const image of images) {
    const code = codeFor(image);
    if (!groups.has(code)) groups.set(code, []);
    groups.get(code).push(image);
  }
  return [...groups].sort(([left], [right]) => left.localeCompare(right));
}

function select(groups, promptId) {
  const rows = groups.map(([code, images]) => {
    const score = keyedScore(code);
    return { code, images, score, race: Math.log(score) / images.length };
  });
  const best = Math.max(...rows.map((row) => row.race));
  const tied = rows.filter((row) => row.race === best);
  const winner = tied[Math.floor(uniform(state.samplerSeed, 0x54100 + promptId) * tied.length)];
  const index = Math.floor(uniform(state.samplerSeed, 0x7ae00 + promptId) * winner.images.length);
  return { rows, winner, image: winner.images[index] };
}

function detect(image) {
  const code = codeFor(image);
  const codes = [code];
  margins(image).forEach((margin, index) => {
    if (Math.abs(margin) < marginLimit) {
      const flipped = code[index] === "0" ? "1" : "0";
      codes.push(code.slice(0, index) + flipped + code.slice(index + 1));
    }
  });
  return { codes, score: Math.max(...codes.map(keyedScore)) };
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function calibrate(prompts) {
  const clean = prompts.map((prompt) => imagesFor(prompt).map((image) => detect(image).score));
  // Exact clean distribution for these four pools. Each clean image is drawn uniformly.
  let sums = [0];
  for (const scores of clean) sums = sums.flatMap((sum) => scores.map((score) => sum + score));
  sums.sort((left, right) => left - right);
  return sums[Math.floor(exampleQuantile * sums.length)] / prompts.length;
}

function svgNode(tag, attributes, parent) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) el.setAttribute(name, value);
  parent.append(el);
  return el;
}

function renderPlanePreview(angles) {
  const preview = byId("plane-preview");
  preview.hidden = !angles;
  if (!angles) return;

  preview.replaceChildren();
  const svg = svgNode("svg", { viewBox: "0 0 32 32", width: 32, height: 32 }, preview);
  svg.setAttribute("aria-hidden", "true");
  for (const angle of angles) {
    const dx = 13 * Math.sin(angle);
    const dy = 13 * Math.cos(angle);
    svgNode("line", { x1: 16 - dx, y1: 16 - dy, x2: 16 + dx, y2: 16 + dy }, svg);
  }
  preview.append(node("span", "", `${angles.length} planes`));
}

function makePlot(images, boundaryAware = false) {
  const plot = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  plot.setAttribute("class", "plot");
  plot.setAttribute("viewBox", "0 0 180 180");
  plot.setAttribute("role", "img");
  plot.setAttribute("aria-label", images.map((image, index) => {
    const codes = boundaryAware ? detect(image).codes : [codeFor(image)];
    return `Image ${index + 1}, ${codes.length > 1 ? "codes" : "code"} ${codes.join(" or ")}`;
  }).join(". "));
  svgNode("circle", { cx: 90, cy: 90, r: 83 }, plot);
  const clipId = boundaryAware ? "detection-disk" : "generation-disk";
  const clip = svgNode("clipPath", { id: clipId }, plot);
  svgNode("circle", { cx: 90, cy: 90, r: 83 }, clip);
  const lines = svgNode("g", { "clip-path": `url(#${clipId})` }, plot);
  for (const [index, angle] of state.angles.entries()) {
    const dx = 83 * Math.sin(angle);
    const dy = 83 * Math.cos(angle);
    const ends = { x1: 90 - dx, y1: 90 - dy, x2: 90 + dx, y2: 90 + dy };
    if (boundaryAware) svgNode("line", { ...ends, class: "band", "stroke-width": 2 * 83 * marginLimit }, lines);
    svgNode("line", ends, lines);
    const labelX = 90 - dx * .88 + Math.cos(angle) * 6;
    const labelY = 90 - dy * .88 - Math.sin(angle) * 6;
    const label = svgNode("text", { x: labelX, y: labelY, "text-anchor": "middle", class: "plane-label" }, plot);
    label.textContent = index + 1;
  }
  const positions = [];
  for (const [index, image] of images.entries()) {
    const x = 90 + 83 * image.x;
    const y = 90 - 83 * image.y;
    const code = codeFor(image);
    const label = boundaryAware ? `#${index + 1}` : `${index + 1} · ${code}`;
    const width = boundaryAware ? 15 : 31;
    const slots = [[0, 0], [0, -23], [0, 23], [-23, 0], [23, 0], [-24, -24], [24, 24], [0, 44], [0, -44]];
    const targets = slots.map(([dx, dy]) => ({
      x: Math.max(width / 2, Math.min(180 - width / 2, x + dx)),
      y: Math.max(14, Math.min(161, y + dy)),
    }));
    const card = targets.find((point) => positions.every((placed) =>
      Math.abs(point.x - placed.x) >= width || Math.abs(point.y - placed.y) >= 22
    )) ?? targets[0];
    positions.push(card);
    if (card.x !== x || card.y !== y) {
      svgNode("line", { x1: x, y1: y, x2: card.x, y2: card.y }, plot);
      svgNode("circle", { cx: x, cy: y, r: 2 }, plot);
    }
    svgNode("image", {
      x: card.x - 9, y: card.y - 9, width: 18, height: 18,
      href: imageRoot + image.file, "data-image": image.file,
    }, plot);
    const text = svgNode("text", { x: card.x, y: card.y + 16, "text-anchor": "middle" }, plot);
    text.textContent = label;
  }
  return plot;
}

function renderGroups(groups, selected) {
  const list = node("div", "groups");
  for (const [code, images] of groups) {
    const bin = node("div", `code-bin${selected?.winner.code === code ? " winner" : ""}`);
    bin.style.setProperty("--code-color", `hsl(${(parseInt(code, 2) * 59 + 215) % 360} 40% 47%)`);
    const label = node("div", "bin-label");
    label.append(node("code", "", code), node("span", "", `${images.length} ${images.length === 1 ? "image" : "images"}`));
    if (selected) label.append(node("span", "key-score", `Key score ${keyedScore(code).toFixed(2)}`));
    bin.append(label, wrapImages(images, (image) => `#${image.index + 1}`));
    list.append(bin);
  }
  return list;
}

function renderSelection(selected) {
  const box = node("div", "selection-result");
  box.append(
    node("strong", "", `Winning code ${selected.winner.code}`),
    makeTile(selected.image, `Returned image · #${selected.image.index + 1}`, "selected-image"),
  );
  return box;
}

function detectionPrompts() {
  const index = labPrompts.indexOf(state.prompt);
  return [state.prompt, ...[4, 8, 12].map((offset) => labPrompts[(index + offset) % labPrompts.length])];
}

function renderScores(batch, results) {
  const list = node("div", "score-list");
  results.forEach(({ codes, score }, index) => {
    const row = node("div", "score-row");
    const leading = node("span", "batch-image");
    const image = node("img");
    image.src = imageRoot + batch[index].file;
    image.alt = "";
    image.dataset.image = batch[index].file;
    leading.append(image, node("span", "", `#${index + 1}`));
    row.append(leading, node("code", "", codes.join(" / ")), node("strong", "", score.toFixed(3)));
    list.append(row);
  });
  return list;
}

function renderVerdict(results, threshold) {
  const score = mean(results.map((item) => item.score));
  const detected = score > threshold;
  const box = node("div", "batch-result");
  const track = node("span", "batch-track");
  track.setAttribute("aria-hidden", "true");
  const bar = node("span");
  bar.style.width = `${score * 100}%`;
  const tick = node("i");
  tick.style.left = `${threshold * 100}%`;
  track.append(bar, tick);
  box.append(
    track,
    node("strong", "", detected ? "Watermark detected" : "Watermark not detected"),
    node("p", "", `Batch average ${score.toFixed(3)} · clean threshold ${threshold.toFixed(3)}`),
    node("p", "", "Demo cutoff at the 80th percentile of unwatermarked draws from the same pools. The score must exceed it."),
  );
  return box;
}

function animateFrom(previousImages, target) {
  if (reducedMotion.matches || !target.animate) return;
  const images = target.querySelectorAll("[data-image]");
  if (!images.length) {
    const contents = target.querySelector(".station-body");
    activeAnimations.push(contents.animate(
      [{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }],
      { duration: 280, easing: "ease-out" },
    ));
  }
  for (const image of images) {
    const start = previousImages.get(image.dataset.image);
    const end = image.getBoundingClientRect();
    if (!end.width) continue;
    if (!start?.width) {
      activeAnimations.push(image.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260 }));
      continue;
    }
    const copy = node("img", "moving-tile");
    copy.src = imageRoot + image.dataset.image;
    copy.alt = "";
    copy.style.width = `${end.width}px`;
    copy.style.height = `${end.height}px`;
    copy.style.left = `${end.left}px`;
    copy.style.top = `${end.top}px`;
    document.body.append(copy);
    movingTiles.push(copy);
    const duration = 480;
    const transform = `translate(${start.left - end.left}px, ${start.top - end.top}px) scale(${start.width / end.width}, ${start.height / end.height})`;
    const transfer = copy.animate(
      [{ transform }, { transform: "none" }],
      { duration, easing: "cubic-bezier(.15,.74,.22,1)" },
    );
    activeAnimations.push(transfer, image.animate(
      [{ opacity: 0 }, { opacity: 0, offset: .96 }, { opacity: 1 }],
      { duration },
    ));
    transfer.addEventListener("finish", () => copy.remove(), { once: true });
  }
}

function renderPipeline(panel, stages, current, views, animate) {
  const previousImages = new Map();
  if (animate) {
    const oldStation = panel.querySelector("[data-current]");
    for (const image of oldStation?.querySelectorAll("[data-image]") ?? []) {
      previousImages.set(image.dataset.image, image.getBoundingClientRect());
    }
  }

  const stations = [...panel.querySelectorAll("[data-station]")];
  stations.forEach((station, index) => {
    const active = index === current;
    station.toggleAttribute("data-current", active);
    if (active) station.setAttribute("aria-current", "step");
    else station.removeAttribute("aria-current");
    station.querySelector("h3").textContent = stages[index];
    station.querySelector(".station-body").replaceChildren(...(index <= current ? [views[index]()] : []));
  });
  if (animate && current >= 0) animateFrom(previousImages, stations[current]);
}

function renderGeneration(animate) {
  const { step, prompt, key, angles } = state;
  const images = candidates(prompt);
  const groups = step >= 2 ? groupImages(images) : [];
  const selected = step === 3 ? select(groups, prompt.id) : null;
  byId("step-count").textContent = `${step + 1} / ${generationActions.length}`;
  byId("calculation").hidden = !selected;
  if (selected) {
    byId("race-values").textContent = selected.rows
      .map(({ code, race }) => `${code}: ${race.toFixed(3)}`).join(" · ");
  }
  previous.disabled = step === 0;
  next.textContent = step === 2 && key === null ? "Draw secret key" : generationActions[step];

  renderPipeline(byId("watermarking"), generationStages, stationForStep[step], [
    () => wrapImages(images, (image) => `#${image.index + 1}`),
    () => step === 1 ? makePlot(images) : renderGroups(groups, selected),
    () => renderSelection(selected),
  ], animate);
  status.textContent = step === 2 && groups.length === 1
    ? "These images all share one code. Try drawing new hyperplanes or choosing another prompt to see multiple groups."
    : step === 1
      ? `${angles.length} hyperplanes give each image a ${angles.length}-bit code. Each line splits the plot into a 0 side and a 1 side.`
      : stepNotes[step];
}

function renderDetection(animate) {
  const step = state.detectionStep;
  const started = step >= 0;
  byId("detection-count").textContent = started ? `${step + 1} / ${detectionActions.length}` : "";
  detectionPrevious.disabled = step <= 0;
  detectionNext.textContent = started ? detectionActions[step] : "Start detection";
  byId("detection-status").textContent = started ? stepNotes[step + 4] : "Use the same key and hyperplanes to check a batch of four images.";

  const prompts = started ? detectionPrompts() : [];
  const batch = prompts.map((prompt) => select(groupImages(candidates(prompt)), prompt.id).image);
  const results = batch.map(detect);
  renderPipeline(byId("detection"), detectionStages, started ? stationForStep[step] : -1, [
    () => wrapImages(batch, (_, index) => `Image ${index + 1}`),
    () => step === 1 ? makePlot(batch, true) : renderScores(batch, results),
    () => renderVerdict(results, calibrate(prompts)),
  ], animate);
}

function render(animatedPanel) {
  activeAnimations.splice(0).forEach((animation) => animation.cancel());
  movingTiles.splice(0).forEach((tile) => tile.remove());
  const { prompt, key, angles } = state;
  byId("prompt-name").textContent = prompt.text;
  byId("prompt-preview").src = imageRoot + imagesFor(prompt)[0].file;
  byId("key-value").textContent = key === null ? "" : `0x${key.toString(16).padStart(8, "0")}`;
  byId("draw-key").textContent = key === null ? "Draw secret key" : "Redraw secret key";
  renderPlanePreview(angles);
  byId("draw-planes").textContent = angles ? "Redraw hyperplanes" : "Draw hyperplanes";
  renderGeneration(animatedPanel === "watermarking");
  renderDetection(animatedPanel === "detection");
}

function drawKey() {
  state.key = randomSeed();
  state.samplerSeed = randomSeed();
}

function drawPlanes() {
  const count = Number(byId("plane-count").value);
  const seeds = crypto.getRandomValues(new Uint32Array(count + 2));
  const turn = seeds[0] / 4294967296 * Math.PI;
  // Space the lines around the plot so each is visible. The frame and bit orientations are random.
  state.angles = Array.from({ length: count }, (_, index) => {
    const jitter = (seeds[index + 2] / 4294967296 - .5) * .55;
    const sign = (seeds[1] >> index) & 1;
    return turn + (index + jitter) * Math.PI / count + sign * Math.PI;
  });
}

function updateKey() {
  drawKey();
  if (state.step === 3) state.step = 2;
  state.detectionStep = -1;
  render();
}

function updatePlanes() {
  drawPlanes();
  state.step = 1;
  state.detectionStep = -1;
  render("watermarking");
}

function resetWalkthroughs() {
  state.step = 0;
  state.detectionStep = -1;
  render();
}

byId("draw-key").addEventListener("click", updateKey);
byId("draw-planes").addEventListener("click", updatePlanes);
byId("plane-count").addEventListener("change", () => {
  if (state.angles) updatePlanes();
});
previous.addEventListener("click", () => {
  state.step -= 1;
  render("watermarking");
});
next.addEventListener("click", () => {
  if (state.step === 0 && !state.angles) {
    updatePlanes();
  } else if (state.step === 2 && state.key === null) {
    updateKey();
    status.textContent = "Secret key drawn. Use it to score the code groups and select an image.";
  } else {
    state.step = (state.step + 1) % generationActions.length;
    render("watermarking");
  }
});
detectionPrevious.addEventListener("click", () => {
  state.detectionStep -= 1;
  render("detection");
});
detectionNext.addEventListener("click", () => {
  if (state.key === null) drawKey();
  if (!state.angles) drawPlanes();
  state.detectionStep = (state.detectionStep + 1) % detectionActions.length;
  render("detection");
});

function renderPromptOptions() {
  const options = labPrompts.map((prompt) => {
    const button = node("button", "prompt-option control");
    button.type = "button";
    button.setAttribute("aria-pressed", String(state.prompt === prompt));
    const preview = node("img");
    preview.src = imageRoot + imagesFor(prompt)[0].file;
    preview.alt = "";
    preview.loading = "lazy";
    button.append(preview, node("span", "", prompt.text));
    button.addEventListener("click", () => {
      state.prompt = prompt;
      dialog.close();
      resetWalkthroughs();
    });
    return button;
  });
  promptGrid.replaceChildren(...options);
}

model.addEventListener("change", resetWalkthroughs);
byId("open-prompts").addEventListener("click", () => {
  renderPromptOptions();
  dialog.showModal();
  promptGrid.querySelector('[aria-pressed="true"]').focus();
});
render();

const copy = document.querySelector("#copy-citation");
const citation = document.querySelector("#citation-text");
copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copy.textContent = "Copied";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copy.textContent = "Text selected";
  }
});
