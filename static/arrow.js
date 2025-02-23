// arrow.js

document.addEventListener("DOMContentLoaded", () => {
    // Track which section is currently "active" based on IntersectionObserver
    let currentSectionIndex = 0;
    const sections = document.querySelectorAll("main section");
  
    // Observer triggers when 60% of a section is visible
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const index = Array.from(sections).indexOf(entry.target);
          if (index !== -1) {
            currentSectionIndex = index;
          }
        }
      });
    }, { threshold: 0.6 });
  
    sections.forEach(section => observer.observe(section));
  
    // Sticky arrow scroll-to-next-section logic
    const arrow = document.getElementById("stickyArrow");
    if (arrow) {
      arrow.addEventListener("click", () => {
        const nextIndex = currentSectionIndex + 1;
        if (nextIndex >= sections.length) {
          // If desired, uncomment to loop back to the first section:
          // sections[0].scrollIntoView({ behavior: "smooth" });
          return;
        }
        sections[nextIndex].scrollIntoView({ behavior: "smooth" });
      });
    }
  
    // Hide the arrow when near the bottom of the page to avoid covering the footer.
    window.addEventListener("scroll", () => {
      if (!arrow) return;
      // Adjust threshold (50px) as needed.
      if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 50) {
        arrow.style.opacity = "0";
        arrow.style.pointerEvents = "none";
      } else {
        arrow.style.opacity = "1";
        arrow.style.pointerEvents = "auto";
      }
    });
  });
  