const animation = document.querySelector("#animation");
const start = document.querySelector("#start");
const back = document.querySelector("#back");
const next = document.querySelector("#next");
const status = document.querySelector("#step-status");
const stages = [
  { label: "Sample several images" },
  { label: "Map images to semantic space", selector: ".a" },
  { label: "Split the space with random hyperplanes", selector: ".b" },
  { label: "Give each region a code", selector: ".c, .ring" },
  { label: "Sample a region from an appropriate distribution", selector: ".d" },
  { label: "The key and group sizes pick region 10", selector: ".e" },
  { label: "Release one image unchanged", selector: ".f" },
  { label: "Embed each image and find its region", selector: ".g" },
  { label: "Score its region and compare to a threshold", selector: ".h" },
];
let layers;
let current = 0;

function showStep(index) {
  current = index;
  layers.forEach((nodes, position) => {
    for (const node of nodes) node.style.visibility = position <= index ? "visible" : "hidden";
  });
  start.disabled = index === 0;
  back.disabled = index === 0;
  next.disabled = index === stages.length - 1;
  status.textContent = `Step ${index + 1} of ${stages.length} · ${stages[index].label}`;
}

function initializePlayer() {
  const diagram = animation.getSVGDocument();
  if (layers || !diagram || diagram.readyState !== "complete") return;
  // Each stage may span several groups in the diagram.
  layers = stages.map(({ selector }) =>
    selector ? [...diagram.querySelectorAll(selector)] : [diagram.documentElement]);
  for (const node of layers.flat()) {
    node.style.animation = "none";
  }
  diagram.querySelectorAll(".ring").forEach((ring) => (ring.style.strokeOpacity = 1));
  showStep(0);
  document.querySelector(".controls").hidden = false;
}

animation.addEventListener("load", initializePlayer);
initializePlayer();
start.addEventListener("click", () => showStep(0));
back.addEventListener("click", () => showStep(current - 1));
next.addEventListener("click", () => showStep(current + 1));
