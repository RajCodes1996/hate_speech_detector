document.addEventListener("DOMContentLoaded", () => {
  const overlay = document.querySelector(".loading-overlay");
  const analyzeForm = document.querySelector("form[data-analyze-form='true']");

  if (!overlay || !analyzeForm) {
    return;
  }

  analyzeForm.addEventListener("submit", () => {
    const button = analyzeForm.querySelector("button[type='submit']");
    if (button) {
      button.disabled = true;
      button.dataset.originalText = button.textContent;
      button.textContent = "Analyzing...";
    }

    window.setTimeout(() => {
      overlay.hidden = false;
      document.body.classList.add("is-loading");
    }, 120);
  });
});
