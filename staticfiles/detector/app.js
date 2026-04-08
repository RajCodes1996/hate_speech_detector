document.addEventListener("DOMContentLoaded", () => {
  const overlay = document.querySelector(".loading-overlay");
  const analyzeForm = document.querySelector("form[data-analyze-form='true']");
  const themeToggle = document.getElementById("theme-toggle");
  const resultModal = document.querySelector("[data-result-modal='true']");
  const resultDialog = resultModal?.querySelector(".result-modal__dialog");
  const resultCloseButtons = resultModal ? resultModal.querySelectorAll("[data-result-close='true']") : [];

  // Initial theme logic
  const currentTheme = localStorage.getItem("theme") || 
      (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? "dark" : "light");
  
  document.documentElement.setAttribute("data-theme", currentTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      let theme = document.documentElement.getAttribute("data-theme");
      let targetTheme = theme === "dark" ? "light" : "dark";
      
      document.documentElement.setAttribute("data-theme", targetTheme);
      localStorage.setItem("theme", targetTheme);
    });
  }

  const openResultModal = () => {
    if (!resultModal || !resultDialog) {
      return;
    }

    resultModal.hidden = false;
    resultModal.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
    window.requestAnimationFrame(() => resultDialog.focus());
  };

  const closeResultModal = () => {
    if (!resultModal) {
      return;
    }

    resultModal.hidden = true;
    resultModal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("modal-open");
  };

  if (resultModal) {
    openResultModal();

    const predictionField = analyzeForm?.querySelector("textarea");
    if (predictionField) {
      predictionField.value = "";
    }

    resultCloseButtons.forEach((button) => {
      button.addEventListener("click", closeResultModal);
    });

    resultModal.addEventListener("click", (event) => {
      if (event.target instanceof HTMLElement && event.target.dataset.resultClose === "true") {
        closeResultModal();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !resultModal.hidden) {
        closeResultModal();
      }
    });
  }

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
