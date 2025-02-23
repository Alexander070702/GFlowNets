"use strict";
// main.js

// HTML Elements
const canvas = document.getElementById("tetrisCanvas");
const ctx = canvas.getContext("2d");
const candidateListEl = document.getElementById("candidateList");
const resetBtn = document.getElementById("resetBtn");


// Gameplay / Visual configs
const CELL_SIZE = 30;
const COLS = 10;
const ROWS = 20;
const TICK_INTERVAL = 700;         // moderate gravity interval
const MOVE_PAUSE_DURATION = 2000;  // 2 seconds after a move is chosen

// Internal state
let currentGameState = null;
let currentPieceCenter = { x: 0, y: 0 };
let candidateMoves = [];
let topCandidates = [];
let appliedArrows = [];
let particles = [];
let totalReward = 0;

let simulationPaused = false;

// Instead of lastPieceType, track piece_id to detect new piece spawns
let lastPieceId = null;
let particleSpawnAccumulator = 0;

//---------------------------
// Utility
//---------------------------
function hexToRgb(hex) {
  hex = hex.replace(/^#/, "");
  const bigint = parseInt(hex, 16);
  return {
    r: (bigint >> 16) & 255,
    g: (bigint >> 8) & 255,
    b: bigint & 255
  };
}

// Assign distinct colors for top 3
function assignCandidateColors(candidates) {
  // #1 green, #2 yellow, #3 red
  candidates.forEach((cand, i) => {
    if (i === 0) cand.color = "#33ff66";   // green (best)
    else if (i === 1) cand.color = "#ffd700"; // gold/yellow
    else if (i === 2) cand.color = "#ff6666"; // light red
    else cand.color = "#dddddd";
  });
}

//---------------------------
// Drawing
//---------------------------
function drawBoard(gs) {
  // Clear entire canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Dark background
  ctx.fillStyle = "#222";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!gs || !gs.board) return;

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (gs.board[r][c]) {
        ctx.fillStyle = "#666"; 
        ctx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);

        ctx.strokeStyle = "#444";
        ctx.strokeRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      } else {
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.strokeRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }
    }
  }
}

function drawCurrentPiece(gs) {
  if (!gs || !gs.current_piece) return;
  const piece = gs.current_piece;

  // Gradient from a light gray to a slightly darker gray
  let grad = ctx.createLinearGradient(0, 0, 0, piece.shape.length * CELL_SIZE);
  grad.addColorStop(0, "#c0c0c0"); // light gray
  grad.addColorStop(1, "#a0a0a0"); // darker gray
  ctx.fillStyle = grad;

  for (let r = 0; r < piece.shape.length; r++) {
    for (let c = 0; c < piece.shape[r].length; c++) {
      if (piece.shape[r][c]) {
        const x = (piece.x + c) * CELL_SIZE;
        const y = (piece.y + r) * CELL_SIZE;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        // Softer outline
        ctx.strokeStyle = "#888";
        ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);
      }
    }
  }
}

function drawCandidateShadow(piece, color) {
  if (!piece || !piece.shape) return;
  ctx.save();
  let rgb = hexToRgb(color || "#ffffff");
  ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.2)`;

  for (let r = 0; r < piece.shape.length; r++) {
    for (let c = 0; c < piece.shape[r].length; c++) {
      if (piece.shape[r][c]) {
        const x = (piece.x + c) * CELL_SIZE;
        const y = (piece.y + r) * CELL_SIZE;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
      }
    }
  }
  ctx.restore();
}

//---------------------------
// Particle + Arrow classes
//---------------------------
class Particle {
  constructor(x, y, vx, vy, radius = 4, life = 1.0, color) {
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.radius = radius;
    this.life = life;
    this.color = color || { r: 255, g: 255, b: 255 };
  }
  update(dt) {
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.life -= dt * 0.4; // fade out
  }
  draw(ctx) {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    let grad = ctx.createRadialGradient(
      this.x, this.y, this.radius / 2,
      this.x, this.y, this.radius
    );
    grad.addColorStop(0, `rgba(${this.color.r},${this.color.g},${this.color.b},${this.life})`);
    grad.addColorStop(1, `rgba(${this.color.r},${this.color.g},${this.color.b},0)`);
    ctx.fillStyle = grad;
    ctx.fill();
  }
}

class Arrow {
  constructor(from, to, flow, color) {
    this.from = from;
    this.to = to;
    this.flow = flow;
    this.color = color || "#66ff66";
    this.life = 1.0;
  }
  update(dt) {
    this.life -= dt * 0.5;
  }
  draw(ctx) {
    let rgb = hexToRgb(this.color);
    // Map flow => line width
    let lineWidth = Math.min(10, 2 + this.flow / 2000);

    ctx.strokeStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${0.8 * this.life})`;
    ctx.lineWidth = lineWidth;

    ctx.beginPath();
    ctx.moveTo(this.from.x, this.from.y);
    ctx.lineTo(this.to.x, this.to.y);
    ctx.stroke();

    // arrowhead
    let angle = Math.atan2(this.to.y - this.from.y, this.to.x - this.from.x);
    ctx.beginPath();
    ctx.moveTo(this.to.x, this.to.y);
    ctx.lineTo(
      this.to.x - 10 * Math.cos(angle - Math.PI / 6),
      this.to.y - 10 * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      this.to.x - 10 * Math.cos(angle + Math.PI / 6),
      this.to.y - 10 * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${0.8 * this.life})`;
    ctx.fill();
  }
}

//---------------------------
// Effects
//---------------------------
function spawnParticles(from, to, factor = 1, color) {
  let count = Math.max(1, Math.round(7 * factor));
  let dx = to.x - from.x;
  let dy = to.y - from.y;
  let dist = Math.hypot(dx, dy) || 1;
  let ux = dx / dist, uy = dy / dist;

  let baseColor = color ? hexToRgb(color) : { r: 255, g: 255, b: 255 };

  for (let i = 0; i < count; i++) {
    const t = Math.random();
    const x = from.x + dx * t;
    const y = from.y + dy * t;

    // random speed
    let speed = 30 + Math.random() * 20;
    let vx = ux * speed + (Math.random() - 0.5) * 10;
    let vy = uy * speed + (Math.random() - 0.5) * 10;
    particles.push(new Particle(x, y, vx, vy, 4, 1.0, baseColor));
  }
}

function spawnArrow(from, to, flow, color) {
  let arrow = new Arrow(from, to, flow, color);
  appliedArrows.push(arrow);
}

//---------------------------
// Render / Animation
//---------------------------
function drawEffects() {
  appliedArrows.forEach(a => a.draw(ctx));
  particles.forEach(p => p.draw(ctx));
}

function draw() {
  drawBoard(currentGameState);
  drawCurrentPiece(currentGameState);

  topCandidates.forEach(c => {
    let arr = new Arrow(currentPieceCenter, c.piece_center, c.flow, c.color);
    arr.draw(ctx);
    drawCandidateShadow(c.piece, c.color);
  });

  drawEffects();
}

let lastTime = performance.now();
function animate() {
  let now = performance.now();
  let dt = (now - lastTime) / 1000;
  lastTime = now;

  // Particle spawns every 0.25s
  particleSpawnAccumulator += dt;
  if (particleSpawnAccumulator > 0.25) {
    if (topCandidates.length > 0) {
      let maxFlow = topCandidates[0].flow || 1;
      topCandidates.forEach((cand, i) => {
        let ratio = cand.flow / maxFlow;
        // #1 => triple, #2 => 1.5, #3 => 0.5
        if (i === 0) ratio *= 3;
        else if (i === 1) ratio *= 1.5;
        else if (i === 2) ratio *= 0.5;
        spawnParticles(currentPieceCenter, cand.piece_center, ratio, cand.color);
      });
    }
    particleSpawnAccumulator = 0;
  }

  // update existing effects
  for (let i = particles.length - 1; i >= 0; i--) {
    particles[i].update(dt);
    if (particles[i].life <= 0) {
      particles.splice(i, 1);
    }
  }
  appliedArrows.forEach(a => a.update(dt));

  draw();
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

//---------------------------
// Server Communication
//---------------------------
function fetchCandidateMoves() {
  if (simulationPaused) return;

  fetch("/api/terminal_moves")
    .then(res => res.json())
    .then(data => {
      currentGameState = data.game_state || {};
      currentPieceCenter = data.current_piece_center || { x:0, y:0 };
      candidateMoves = data.terminal_moves || [];

      candidateMoves.sort((a, b) => b.flow - a.flow);
      topCandidates = candidateMoves.slice(0, 3);
      assignCandidateColors(topCandidates);

      updateCandidateList();


      // Auto-click the first candidate in the list if it exists
      const firstCandidate = candidateListEl.querySelector(".candidate");
      if (firstCandidate) {
        firstCandidate.click();  // triggers the .onclick from updateCandidateList
      }
    })
    .catch(err => console.error(err));
}

function selectCandidate(actionKey) {
  simulationPaused = true;

  let payload = actionKey ? { action_key: actionKey } : {};
  fetch("/api/select_move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        console.error("select_move error:", data.error);
        simulationPaused = false;
        return;
      }
      if (data.arrow) {
        let arrowData = data.arrow;
        let cand = topCandidates.find(
          c => Math.abs(c.piece_center.x - arrowData.to.x) < 1 &&
               Math.abs(c.piece_center.y - arrowData.to.y) < 1
        );
        let color = cand ? cand.color : "#33ff66";
        spawnArrow(arrowData.from, arrowData.to, arrowData.flow, color);
      }

      currentGameState = data.game_state || {};


      // Wait a short period to let the move animate before unpausing
      setTimeout(() => {
        simulationPaused = false;
      }, MOVE_PAUSE_DURATION);
    })
    .catch(err => {
      console.error(err);
      simulationPaused = false;
    });
}

function tickGame() {
  if (!simulationPaused) {
    fetch("/api/tick", { method: "POST" })
      .then(res => res.json())
      .then(data => {
        currentGameState = data.game_state || {};
        currentPieceCenter = data.current_piece_center || { x:0, y:0 };

        // Grab piece_id from the server
        let newPieceId = currentGameState.piece_id;
        // If piece_id changed => truly new piece => fetch new terminal states
        if (typeof newPieceId !== "undefined" && newPieceId !== lastPieceId) {
          lastPieceId = newPieceId;
          // slight delay to let the piece spawn
          setTimeout(fetchCandidateMoves, 200);
        }
      })
      .catch(err => console.error(err));
  }
}

//---------------------------
// UI
//---------------------------
function updateCandidateList() {
  candidateListEl.innerHTML = "";
  topCandidates.forEach(c => {
    let div = document.createElement("div");
    div.className = "candidate";
    div.style.borderLeft = `10px solid ${c.color}`;
    div.innerHTML = `
      <h3>${c.action_key}</h3>
      <p>Flow: ${c.flow.toFixed(2)}</p>
      <p>Prob: ${(c.probability * 100).toFixed(1)}%</p>
    `;
    div.onclick = () => selectCandidate(c.action_key);
    candidateListEl.appendChild(div);
  });
}

resetBtn.addEventListener("click", () => {
  fetch("/api/reset", { method: "POST" })
    .then(res => res.json())
    .then(() => {
      // Clear everything
      currentGameState = null;
      currentPieceCenter = { x: 0, y: 0 };
      candidateMoves = [];
      topCandidates = [];
      appliedArrows = [];
      particles = [];
      simulationPaused = false;
      lastPieceId = null;
      candidateListEl.innerHTML = "";

      fetchCandidateMoves();
    })
    .catch(err => console.error(err));
});

//---------------------------
// Init
//---------------------------
setInterval(tickGame, TICK_INTERVAL);
fetchCandidateMoves();
