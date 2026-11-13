const demo = document.querySelector("#walkthrough");
const steps = [...demo.querySelectorAll("[data-panel]")];
const stepButtons = [...demo.querySelectorAll("[data-step]")];
const previous = demo.querySelector("[data-previous]");
const next = demo.querySelector("[data-next]");
const play = demo.querySelector("[data-play]");
const status = demo.querySelector("[data-status]");
const interval = 7000;
let current = 0;
let timer;

function showStep(index) {
  current = index;
  steps.forEach((step, position) => {
    step.hidden = position !== index;
  });
  stepButtons.forEach((button, position) => {
    button.setAttribute("aria-pressed", String(position === index));
  });
  previous.disabled = index === 0;
  next.disabled = index === steps.length - 1;
  status.textContent = `Step ${index + 1} of ${steps.length} · ${stepButtons[index].textContent.trim()}`;
  play.textContent = timer ? "Pause" : index === steps.length - 1 ? "Replay" : "Play";
}

function stop() {
  clearInterval(timer);
  timer = undefined;
}

function jump(index) {
  stop();
  showStep(index);
}

stepButtons.forEach((button, index) => button.addEventListener("click", () => jump(index)));
previous.addEventListener("click", () => jump(current - 1));
next.addEventListener("click", () => jump(current + 1));
play.addEventListener("click", () => {
  if (timer) {
    jump(current);
    return;
  }
  const start = current === steps.length - 1 ? 0 : current;
  timer = setInterval(() => {
    if (current === steps.length - 1) {
      jump(current);
    } else {
      showStep(current + 1);
    }
  }, interval);
  showStep(start);
});
showStep(0);

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
