const animation = document.querySelector("#animation");
const start = document.querySelector("#start");
const back = document.querySelector("#back");
const next = document.querySelector("#next");
const status = document.querySelector("#step-status");
const stages = [
  { label: "Generate candidates" },
  { label: "Group and score image codes", selector: ".group" },
  { label: "Return one image", selector: ".pick" },
  { label: "Turn the cropped image into a code", selector: ".scoring" },
  { label: "Compare the score with the threshold", selector: ".decision" },
];
let layers;
let current = 0;

function showStep(index) {
  current = index;
  layers.forEach((layer, position) => {
    layer.style.visibility = position <= index ? "visible" : "hidden";
  });
  start.disabled = index === 0;
  back.disabled = index === 0;
  next.disabled = index === stages.length - 1;
  status.textContent = `Step ${index + 1} of ${stages.length} · ${stages[index].label}`;
}

function initializePlayer() {
  const diagram = animation.getSVGDocument();
  if (layers || !diagram || diagram.readyState !== "complete") return;
  layers = stages.map(({ selector }) =>
    selector ? diagram.querySelector(selector) : diagram.documentElement);
  for (const layer of layers) {
    layer.style.animation = "none";
  }
  showStep(0);
  document.querySelector(".controls").hidden = false;
}

animation.addEventListener("load", initializePlayer);
initializePlayer();
start.addEventListener("click", () => showStep(0));
back.addEventListener("click", () => showStep(current - 1));
next.addEventListener("click", () => showStep(current + 1));
