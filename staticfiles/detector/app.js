document.addEventListener("DOMContentLoaded", () => {
  const overlay = document.querySelector(".loading-overlay");
  const analyzeForm = document.querySelector("form[data-analyze-form='true']");
  const themeToggle = document.getElementById("theme-toggle");

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
