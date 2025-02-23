// nav.js
document.addEventListener("DOMContentLoaded", () => {
    const burger = document.getElementById("burger");
    const navMenu = document.querySelector(".nav-menu");
  
    if (burger && navMenu) {
      burger.addEventListener("click", () => {
        // Toggle the "active" class on the nav menu
        navMenu.classList.toggle("active");
        // Optionally animate the burger icon by toggling a class on it
        burger.classList.toggle("active");
      });
    }
  });
  