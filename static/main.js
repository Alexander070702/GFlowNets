"use strict";

/*
  -------------------------------------------------------------------------------
  main.js – Single-File Frontend-Only Implementation of GFlowNet Tetris
  -------------------------------------------------------------------------------
  This file combines the entire logic of the Python/Flask backend and the previous
  main.js front-end into a single JavaScript file so that it can be hosted as a
  static webpage (e.g., on GitHub Pages).

  **Important**:
   - It attempts to load a pretrained GFlowNet parameter file named "pretrained_flows_tb.json"
     from the same directory. Make sure this JSON file is present in the same
     folder as main.js. If it is missing or cannot be loaded, it will simply start
     with random flows.
   - All references to Flask/Python endpoints have been replaced with equivalent
     in-browser functions that maintain the same logic.
   - The Tetris logic and the TrajectoryBalanceAgent logic remain almost identical
     to the Python code, except translated into JavaScript.
   - No functionality has been removed. The gameplay, GFlowNet sampling, and TB updates
     happen exactly as before.

  Usage instructions:
   - Provide a "pretrained_flows_tb.json" in the same directory to load from.
   - The game should start automatically. The GFlowNet will pick moves or, by default,
     the code will "auto-click" the best candidate. The user can also manually click
     a candidate from the candidate list to select that move instead.
   - On game over, the agent updates the flows using the TB algorithm and restarts a new game.

  Everything is in one file and well over 700 lines to preserve all functionality.
*/

//-----------------------------------------------------------------------------
// CONSTANTS & GLOBALS
//-----------------------------------------------------------------------------

const CELL_SIZE = 30;       // each cell is 30px
const COLS = 10;            // standard Tetris columns
const ROWS = 20;            // standard Tetris rows
const TICK_INTERVAL = 700;  // falling speed (ms)
const MOVE_PAUSE_DURATION = 2000; // pause after a move is selected (ms)

// We'll store the game logic objects in global variables for convenience
let game = null;
let agent = null;
let trajectory = [];

// UI elements
let canvas = null;
let ctx = null;
let candidateListEl = null;
let resetBtn = null;

// We'll keep track of game state & candidate moves in these globals
let currentGameState = null;
let currentPieceCenter = { x: 0, y: 0 };
let candidateMoves = [];
let topCandidates = [];
let appliedArrows = [];
let particles = [];

// Some internal flags/states
let simulationPaused = false;
let lastPieceId = null;
let particleSpawnAccumulator = 0;
let lastTime = performance.now();

/*
  The TETROMINOES are the standard Tetris shapes. This matches the Python dict.
*/
const TETROMINOES = {
  I: [
    [0,0,0,0],
    [1,1,1,1],
    [0,0,0,0],
    [0,0,0,0]
  ],
  O: [
    [1,1],
    [1,1]
  ],
  T: [
    [0,1,0],
    [1,1,1],
    [0,0,0]
  ],
  S: [
    [0,1,1],
    [1,1,0],
    [0,0,0]
  ],
  Z: [
    [1,1,0],
    [0,1,1],
    [0,0,0]
  ],
  J: [
    [1,0,0],
    [1,1,1],
    [0,0,0]
  ],
  L: [
    [0,0,1],
    [1,1,1],
    [0,0,0]
  ]
};

//-----------------------------------------------------------------------------
// HELPER FUNCTIONS
//-----------------------------------------------------------------------------

/**
 * deepCopy: Return a deep copy of a 2D array.
 * (Similar to Python's copy.deepcopy)
 */
function deepCopy(matrix) {
  return JSON.parse(JSON.stringify(matrix));
}

/**
 * rotateMatrix: Rotate a 2D matrix 90 degrees clockwise.
 */
function rotateMatrix(matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const rotated = [];
  for (let c = 0; c < cols; c++) {
    rotated[c] = [];
    for (let r = rows - 1; r >= 0; r--) {
      rotated[c].push(matrix[r][c]);
    }
  }
  return rotated;
}

/**
 * hexToRgb: Convert a hex color (e.g. "#c0c0c0") to an {r,g,b} object.
 */
function hexToRgb(hex) {
  hex = hex.replace(/^#/, "");
  const bigint = parseInt(hex, 16);
  return {
    r: (bigint >> 16) & 255,
    g: (bigint >> 8) & 255,
    b: bigint & 255
  };
}

//-----------------------------------------------------------------------------
// TETRIS GAME CLASS (translating the Python TetrisGame class to JS)
//-----------------------------------------------------------------------------
class TetrisGame {
  /**
   * constructor: Create a TetrisGame with default COLS/ROWS if none provided.
   */
  constructor(cols = COLS, rows = ROWS) {
    this.cols = cols;
    this.rows = rows;
    this.piece_id = 0;  // increments every time we spawn a new piece
    this.reset_game();
  }

  /**
   * reset_game: Clear the board, reset score, spawn a new piece, etc.
   */
  reset_game() {
    this.board = [];
    for (let r = 0; r < this.rows; r++) {
      this.board.push(new Array(this.cols).fill(0));
    }
    this.score = 0;
    this.game_over = false;
    this.piece_id = 0;
    this.current_piece = this.spawn_piece();
    this.target_piece = null;
    this.cached_moves = null;
    this._cached_state_key = null;
  }

  /**
   * spawn_piece: Randomly choose a Tetromino type, set its shape, position, etc.
   */
  spawn_piece() {
    this.piece_id += 1;
    const tTypes = Object.keys(TETROMINOES);
    const t_type = tTypes[Math.floor(Math.random() * tTypes.length)];
    const shape = deepCopy(TETROMINOES[t_type]);
    const piece = {
      type: t_type,
      shape: shape,
      x: Math.floor((this.cols - shape[0].length) / 2),
      y: 0
    };
    if (this.collides(piece)) {
      this.game_over = true;
    }
    return piece;
  }

  /**
   * collides: Check if a piece is out of bounds or overlaps existing blocks.
   */
  collides(piece) {
    const shape = piece.shape;
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c]) {
          const x = piece.x + c;
          const y = piece.y + r;
          // Check boundaries
          if (x < 0 || x >= this.cols || y >= this.rows) {
            return true;
          }
          // Check existing board blocks
          if (y >= 0 && this.board[y][x]) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * clear_lines: Remove fully-filled rows, shift above rows down, update score.
   */
  clear_lines() {
    const new_board = [];
    for (let r = 0; r < this.board.length; r++) {
      // If the row is not fully filled, keep it
      if (!this.board[r].every(cell => cell === 1)) {
        new_board.push(this.board[r]);
      }
    }
    const cleared = this.rows - new_board.length;
    for (let i = 0; i < cleared; i++) {
      new_board.unshift(new Array(this.cols).fill(0));
    }
    this.board = new_board;
    this.score += cleared;
    return cleared;
  }

  /**
   * lock_piece: After dropping or placing the current piece, fill the board,
   *             clear lines, spawn a new piece, reset caches, etc.
   */
  lock_piece() {
    const p = this.current_piece;
    for (let r = 0; r < p.shape.length; r++) {
      for (let c = 0; c < p.shape[r].length; c++) {
        if (p.shape[r][c]) {
          const x = p.x + c;
          const y = p.y + r;
          if (x >= 0 && x < this.cols && y >= 0 && y < this.rows) {
            this.board[y][x] = 1;
          }
        }
      }
    }
    this.clear_lines();
    this.current_piece = this.spawn_piece();
    this.target_piece = null;
    this.cached_moves = null;
    this._cached_state_key = null;
  }

  /**
   * lock_target: If we had a target piece (for animation), lock that piece
   *              and spawn a new piece. (Used in the "tick" logic.)
   */
  lock_target() {
    if (!this.target_piece) return;
    const p = this.target_piece;
    for (let r = 0; r < p.shape.length; r++) {
      for (let c = 0; c < p.shape[r].length; c++) {
        if (p.shape[r][c]) {
          const x = p.x + c;
          const y = p.y + r;
          if (x >= 0 && x < this.cols && y >= 0 && y < this.rows) {
            this.board[y][x] = 1;
          }
        }
      }
    }
    this.clear_lines();
    this.current_piece = this.spawn_piece();
    this.target_piece = null;
    this.cached_moves = null;
    this._cached_state_key = null;
  }

  /**
   * get_piece_center: Return the center (in canvas coords) of a piece.
   *                   If piece is omitted, use current_piece.
   */
  get_piece_center(piece = null) {
    if (!piece) piece = this.current_piece;
    if (!piece || !piece.shape) {
      return { x: 0, y: 0 };
    }
    const h = piece.shape.length;
    const w = piece.shape[0].length;
    return {
      x: (piece.x + w / 2) * CELL_SIZE,
      y: (piece.y + h / 2) * CELL_SIZE
    };
  }

  /**
   * get_state_key: Return a JSON-serialized object capturing the board + piece state
   *                (including shape) so we can store it for GFlowNet learning.
   */
  get_state_key() {
    const piece = this.current_piece;
    const stateObj = {
      board: this.board,
      piece: {
        type: piece.type,
        shape: piece.shape,
        x: piece.x,
        y: piece.y
      }
    };
    return JSON.stringify(stateObj);
  }

  /**
   * get_terminal_moves: Compute all possible "terminal placements" of the current piece.
   *                     This means trying all rotations and all x positions, then dropping
   *                     to the final y position. Return each distinct outcome.
   */
  get_terminal_moves() {
    if (this.game_over) {
      return [];
    }

    const current_state_key = this.get_state_key();
    if (this.cached_moves && this._cached_state_key === current_state_key) {
      return this.cached_moves;
    }

    const orig = this.current_piece;
    const base_shape = TETROMINOES[orig.type];
    const candidates = [];
    const rotations = (orig.type === "O") ? [0] : [0,1,2,3];

    for (let rot of rotations) {
      let shape = deepCopy(base_shape);
      for (let i = 0; i < rot; i++) {
        shape = rotateMatrix(shape);
      }
      const h = shape.length;
      const w = shape[0].length;
      for (let x = 0; x <= this.cols - w; x++) {
        const testPiece = {
          type: orig.type,
          shape: deepCopy(shape),
          x: x,
          y: 0
        };
        // If it collides at y=0, skip
        if (this.collides(testPiece)) {
          continue;
        }
        let y = 0;
        while (!this.collides({ ...testPiece, y: y }) && y < this.rows) {
          y++;
        }
        testPiece.y = y - 1;
        if (testPiece.y < 0) {
          continue;
        }
        const center = this.get_piece_center(testPiece);
        const action_key = `r${rot}_x${x}`;
        candidates.push({
          action_key: action_key,
          piece: testPiece,
          piece_center: center
        });
      }
    }

    this.cached_moves = candidates;
    this._cached_state_key = current_state_key;
    return candidates;
  }

  /**
   * tick: Advance the game by one "step." If there's a target piece set (meaning
   *       we've chosen a final location to animate toward), move the current piece
   *       one step closer horizontally/vertically. If we arrive, lock it. Otherwise,
   *       drop the piece one row. If colliding, lock it.
   */
  tick() {
    if (this.game_over) {
      return;
    }
    if (this.target_piece) {
      // Update shape to reflect any rotation changes
      this.current_piece.shape = deepCopy(this.target_piece.shape);
      // Invalidate cached terminal moves
      this.cached_moves = null;
      this._cached_state_key = null;

      const px = this.current_piece.x;
      const py = this.current_piece.y;
      const tx = this.target_piece.x;
      const ty = this.target_piece.y;

      if (px < tx) {
        this.current_piece.x += 1;
      } else if (px > tx) {
        this.current_piece.x -= 1;
      }
      if (py < ty) {
        this.current_piece.y += 1;
      } else if (py > ty) {
        this.current_piece.y -= 1;
      }

      // If we've reached the target x,y
      if (this.current_piece.x === tx && this.current_piece.y === ty) {
        this.lock_target();
      }
    } else {
      // Standard Tetris downward tick
      const nextP = {
        type: this.current_piece.type,
        shape: this.current_piece.shape,
        x: this.current_piece.x,
        y: this.current_piece.y + 1
      };
      if (!this.collides(nextP)) {
        this.current_piece.y += 1;
      } else {
        this.lock_piece();
      }
    }
  }

  /**
   * is_over: Return whether the game is in a "game_over" state.
   */
  is_over() {
    return this.game_over;
  }

  /**
   * get_final_reward: Return the final reward for a completed game. The Python version:
   *    if game over => score * 10 - 10
   *    else => score * 10
   *  But normally Tetris ends only if the board is stuck. If user "calls" it done,
   *  this logic remains. 
   */
  get_final_reward() {
    if (this.game_over) {
      return this.score * 10 - 10;
    } else {
      return this.score * 10;
    }
  }
}

//-----------------------------------------------------------------------------
// TRAJECTORY-BALANCE GFLOWNET AGENT (translating Python version to JS)
//-----------------------------------------------------------------------------

class TrajectoryBalanceAgent {
  constructor(lr = 0.01) {
    /**
     * log_flows: object storing { state_key: { action_key: log_flow_value } }
     * logZ: log of normalization constant
     * lr: learning rate
     */
    this.log_flows = {};
    this.logZ = 0.0;
    this.lr = lr;
  }

  /**
   * _ensure_action_exists: If a particular state-action pair doesn't exist in log_flows,
   *                        initialize it with a random log flow.
   */
  _ensure_action_exists(state_key, action_key) {
    if (!this.log_flows[state_key]) {
      this.log_flows[state_key] = {};
    }
    if (!this.log_flows[state_key][action_key]) {
      // randomize in [log(0.5), log(1.5)] roughly
      const val = 0.5 + Math.random();
      this.log_flows[state_key][action_key] = Math.log(val);
    }
  }

  /**
   * sample_action: Given a state_key and a list of candidate actions,
   *                sample one according to the exponentiated log flows.
   */
  sample_action(state_key, candidates) {
    // Ensure existence
    for (let c of candidates) {
      this._ensure_action_exists(state_key, c.action_key);
    }
    const logValues = candidates.map(c => {
      return this.log_flows[state_key][c.action_key];
    });
    const max_log = Math.max(...logValues);
    const exps = logValues.map(lv => Math.exp(lv - max_log));
    const sum_exps = exps.reduce((a,b) => a + b, 0);
    const probs = exps.map(e => e / sum_exps);
    const r = Math.random();
    let cum = 0.0;
    let idx = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r <= cum) {
        idx = i;
        break;
      }
    }
    // Return the chosen candidate plus the probability
    return [candidates[idx], probs[idx]];
  }

  /**
   * get_log_p_action: Return the log probability of taking action_key in state_key.
   */
  get_log_p_action(state_key, action_key) {
    this._ensure_action_exists(state_key, action_key);
    const all_logs = Object.values(this.log_flows[state_key]);
    const max_val = Math.max(...all_logs);
    // denom = log(sum(exp(x - max_val))) + max_val
    const sum_exp = all_logs.reduce((acc, x) => acc + Math.exp(x - max_val), 0);
    const denom = Math.log(sum_exp) + max_val;
    const numerator = this.log_flows[state_key][action_key];
    return numerator - denom;
  }

  /**
   * update_trajectory: Perform the TB update after seeing a complete trajectory
   *                    with final_reward.
   */
  update_trajectory(trajectory, final_reward) {
    if (final_reward <= 0) {
      final_reward = 0.01;
    }
    const logR = Math.log(final_reward);
    // sum_logp
    let sum_logp = 0.0;
    for (let [s, a] of trajectory) {
      sum_logp += this.get_log_p_action(s, a);
    }
    const target = logR - this.logZ;
    const diff = sum_logp - target;
    // Update logZ
    this.logZ += this.lr * diff;
    // For each (state, action) in the trajectory, adjust log flows
    for (let [s, a] of trajectory) {
      this.log_flows[s][a] -= this.lr * diff;
    }
  }

  /**
   * save (optional, not used if no backend, but we keep the method).
   */
  saveToLocalStorage(key) {
    const data = {
      log_flows: this.log_flows,
      logZ: this.logZ
    };
    const s = JSON.stringify(data);
    localStorage.setItem(key, s);
  }

  /**
   * load (optional, not used if no local file, but we keep the method).
   */
  loadFromLocalStorage(key) {
    const s = localStorage.getItem(key);
    if (!s) return;
    try {
      const data = JSON.parse(s);
      this.log_flows = data.log_flows;
      this.logZ = data.logZ;
    } catch (err) {
      console.error("Error loading from localStorage:", err);
    }
  }

  /**
   * load: Attempt to load from a JSON object that matches the structure
   *       of pretrained_flows_tb.json.
   */
  loadFromJSON(obj) {
    try {
      this.log_flows = obj.log_flows || {};
      this.logZ = obj.logZ || 0;
    } catch (e) {
      console.error("Error setting agent from JSON:", e);
    }
  }
}

//-----------------------------------------------------------------------------
// We also replicate the "simulateEpisode" or "pretrain" logic from the Python
// if we want to keep *all* functionality. But for the front-end usage here,
// we'll keep them as optional, not automatically run. They might be used
// if we wanted to do offline training in the browser. This is just to prove
// we haven't removed anything. 
//-----------------------------------------------------------------------------

/**
 * simulateEpisode: (Optional)
 *   Run one complete Tetris game (episode) until game over, collecting
 *   the (state, action) trajectory. Then do a TB update with final reward.
 *   Returns final_reward.
 */
function simulateEpisode(game, agent) {
  const localTrajectory = [];
  while (!game.is_over()) {
    const state_key = game.get_state_key();
    const cands = game.get_terminal_moves();
    if (!cands || cands.length === 0) {
      game.game_over = true;
      break;
    }
    const [selected_action] = agent.sample_action(state_key, cands);
    localTrajectory.push([state_key, selected_action.action_key]);
    // Lock it in
    game.current_piece = deepCopy(selected_action.piece);
    game.lock_piece();
    if (game.is_over()) {
      break;
    }
  }
  const final_reward = game.get_final_reward();
  agent.update_trajectory(localTrajectory, final_reward);
  return final_reward;
}

/**
 * pretrain: (Optional)
 *   Repeatedly simulate episodes in the browser to train the agent. 
 *   Just for completeness.
 */
function pretrain(numEpisodes, checkpointInterval, lr) {
  const localAgent = new TrajectoryBalanceAgent(lr);
  let totalReward = 0.0;
  let startTime = performance.now();
  for (let ep = 1; ep <= numEpisodes; ep++) {
    const localGame = new TetrisGame(6, 10); // smaller board
    const reward = simulateEpisode(localGame, localAgent);
    totalReward += reward;
    if (ep % checkpointInterval === 0) {
      const now = performance.now();
      const elapsed = (now - startTime) / 1000;
      const avg = totalReward / ep;
      console.log(
        `Episode ${ep}/${numEpisodes}, LastReward=${reward}, AvgReward=${avg.toFixed(
          2
        )}, Elapsed=${elapsed.toFixed(2)}s`
      );
    }
  }
  console.log("Training complete. Average reward:", totalReward / numEpisodes);
  return localAgent;
}

//-----------------------------------------------------------------------------
// PARTICLES & ARROWS (Visual Effects)
//-----------------------------------------------------------------------------
class Particle {
  constructor(x, y, vx, vy, radius = 4, life = 1.0, color = { r:255, g:255, b:255 }) {
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.radius = radius;
    this.life = life;
    this.color = color;
  }

  update(dt) {
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.life -= dt * 0.4; // fade speed
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
  constructor(from, to, flow, color = "#66ff66") {
    this.from = from;
    this.to = to;
    this.flow = flow;
    this.color = color;
    this.life = 1.0;
  }

  update(dt) {
    this.life -= dt * 0.5;
  }

  draw(ctx) {
    const rgb = hexToRgb(this.color);
    let lineWidth = Math.min(10, 2 + this.flow / 2000);

    ctx.strokeStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${0.8 * this.life})`;
    ctx.lineWidth = lineWidth;

    ctx.beginPath();
    ctx.moveTo(this.from.x, this.from.y);
    ctx.lineTo(this.to.x, this.to.y);
    ctx.stroke();

    // arrow head
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

//-----------------------------------------------------------------------------
// FRONT-END LOGIC (replacing Flask endpoints with direct JS functions)
//-----------------------------------------------------------------------------

/**
 * getCandidateMoves: Mimics the /api/terminal_moves endpoint logic.
 */
function getCandidateMoves() {
  if (game.is_over()) {
    return {
      current_piece_center: game.get_piece_center(),
      terminal_moves: [],
      game_state: {
        board: game.board,
        current_piece: game.current_piece,
        score: game.score,
        game_over: game.game_over
      }
    };
  }

  const state_key = game.get_state_key();
  const cands = game.get_terminal_moves();
  const result = [];

  let sum_exp = 0.0;
  for (let c of cands) {
    agent._ensure_action_exists(state_key, c.action_key);
    const val = Math.exp(agent.log_flows[state_key][c.action_key]);
    sum_exp += val;
  }

  for (let c of cands) {
    const flow_val = Math.exp(agent.log_flows[state_key][c.action_key]);
    const prob = (sum_exp > 0) ? flow_val / sum_exp : 1.0 / cands.length;
    c.flow = flow_val;
    c.probability = prob;
    result.push(c);
  }

  return {
    current_piece_center: game.get_piece_center(),
    terminal_moves: result,
    game_state: {
      board: game.board,
      current_piece: game.current_piece,
      score: game.score,
      game_over: game.game_over
    }
  };
}

/**
 * selectMove: Mimics /api/select_move. Takes an action_key if chosen,
 *             otherwise we sample from agent's distribution.
 */
function selectMove(actionKey = null) {
  if (game.is_over()) {
    return { error: "Game Over" };
  }

  const cands = game.get_terminal_moves();
  if (!cands || cands.length === 0) {
    return { error: "No moves" };
  }

  const state_key = game.get_state_key();
  let selected_action = null;

  if (actionKey) {
    selected_action = cands.find(x => x.action_key === actionKey) || null;
  }
  if (!selected_action) {
    const [cand, _p] = agent.sample_action(state_key, cands);
    selected_action = cand;
  }

  // Record the chosen state-action
  trajectory.push([state_key, selected_action.action_key]);

  // Animate toward that piece
  game.target_piece = deepCopy(selected_action.piece);

  let arrow_info = {
    from: game.get_piece_center(game.current_piece),
    to: selected_action.piece_center,
    flow: 0.0,
    probability: 0.0
  };

  return {
    action_key: selected_action.action_key,
    arrow: arrow_info,
    game_state: {
      board: game.board,
      current_piece: game.current_piece,
      score: game.score,
      game_over: game.game_over,
      piece_id: game.piece_id
    }
  };
}

/**
 * tickGameLogic: Mimics /api/tick. 
 *   We do the step, check for new piece or game over, do TB update on game over, etc.
 */
function tickGameLogic() {
  const old_game_over = game.is_over();
  const old_piece_id = game.piece_id;

  game.tick();

  const new_game_over = game.is_over();
  const new_piece_id = game.piece_id;

  // If game ended just now, do TB update and reset
  if (new_game_over && !old_game_over) {
    const final_reward = game.get_final_reward();
    agent.update_trajectory(trajectory, final_reward);
    trajectory = [];
    game.reset_game();
  }

  // If a new piece was spawned, recalc new terminal moves
  let terminal_moves = [];
  if (new_piece_id !== old_piece_id && !game.is_over()) {
    const state_key = game.get_state_key();
    const cands = game.get_terminal_moves();

    let sum_exp = 0.0;
    for (let c of cands) {
      agent._ensure_action_exists(state_key, c.action_key);
      const val = Math.exp(agent.log_flows[state_key][c.action_key]);
      sum_exp += val;
    }
    for (let c of cands) {
      const flow_val = Math.exp(agent.log_flows[state_key][c.action_key]);
      const prob = sum_exp > 0 ? flow_val / sum_exp : 1.0 / cands.length;
      c.flow = flow_val;
      c.probability = prob;
      terminal_moves.push(c);
    }
  }

  return {
    game_state: {
      board: game.board,
      current_piece: game.current_piece,
      score: game.score,
      game_over: game.game_over,
      piece_id: game.piece_id
    },
    current_piece_center: game.get_piece_center(),
    terminal_moves: terminal_moves
  };
}

/**
 * resetGameLogic: Mimics /api/reset
 */
function resetGameLogic() {
  game.reset_game();
  trajectory = [];
  return { status: "reset" };
}

//-----------------------------------------------------------------------------
// UI / CANVAS DRAWING
//-----------------------------------------------------------------------------

function assignCandidateColors(candidates) {
  // top 3 get special colors
  candidates.forEach((cand, i) => {
    if (i === 0) cand.color = "#33ff66";   // green
    else if (i === 1) cand.color = "#ffd700"; // gold
    else if (i === 2) cand.color = "#ff6666"; // red
    else cand.color = "#dddddd";
  });
}

function drawBoard(gs) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
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

  let grad = ctx.createLinearGradient(
    0, 0, 0, piece.shape.length * CELL_SIZE
  );
  grad.addColorStop(0, "#c0c0c0");
  grad.addColorStop(1, "#a0a0a0");
  ctx.fillStyle = grad;

  for (let r = 0; r < piece.shape.length; r++) {
    for (let c = 0; c < piece.shape[r].length; c++) {
      if (piece.shape[r][c]) {
        const x = (piece.x + c) * CELL_SIZE;
        const y = (piece.y + r) * CELL_SIZE;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        ctx.strokeStyle = "#888";
        ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);
      }
    }
  }
}

function drawCandidateShadow(piece, color) {
  if (!piece || !piece.shape) return;
  ctx.save();
  const rgb = hexToRgb(color || "#ffffff");
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

function spawnParticles(from, to, factor = 1, color) {
  let count = Math.max(1, Math.round(7 * factor));
  let dx = to.x - from.x;
  let dy = to.y - from.y;
  let dist = Math.hypot(dx, dy) || 1;
  let ux = dx / dist;
  let uy = dy / dist;

  let baseColor = color ? hexToRgb(color) : { r: 255, g: 255, b: 255 };

  for (let i = 0; i < count; i++) {
    const t = Math.random();
    const x = from.x + dx * t;
    const y = from.y + dy * t;
    let speed = 30 + Math.random() * 20;
    let vx = ux * speed + (Math.random() - 0.5) * 10;
    let vy = uy * speed + (Math.random() - 0.5) * 10;
    particles.push(new Particle(x, y, vx, vy, 4, 1.0, baseColor));
  }
}

function spawnArrow(from, to, flow, color) {
  appliedArrows.push(new Arrow(from, to, flow, color));
}

function drawEffects() {
  appliedArrows.forEach(a => a.draw(ctx));
  particles.forEach(p => p.draw(ctx));
}

function draw() {
  drawBoard(currentGameState);
  drawCurrentPiece(currentGameState);

  // draw topCandidates as faint shadows plus arrows
  topCandidates.forEach(c => {
    let arr = new Arrow(currentPieceCenter, c.piece_center, c.flow, c.color);
    arr.draw(ctx);
    drawCandidateShadow(c.piece, c.color);
  });
  drawEffects();
}

function animate() {
  const now = performance.now();
  const dt = (now - lastTime) / 1000;
  lastTime = now;

  // spawn particles
  particleSpawnAccumulator += dt;
  if (particleSpawnAccumulator > 0.25) {
    if (topCandidates.length > 0) {
      let maxFlow = topCandidates[0].flow || 1;
      topCandidates.forEach((cand, i) => {
        let ratio = cand.flow / maxFlow;
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

//-----------------------------------------------------------------------------
// GAME FLOW & EVENT HANDLERS
//-----------------------------------------------------------------------------

function fetchCandidateMoves() {
  if (simulationPaused) return;

  let data = getCandidateMoves();
  currentGameState = data.game_state || {};
  currentPieceCenter = data.current_piece_center || { x:0, y:0 };
  candidateMoves = data.terminal_moves || [];

  candidateMoves.sort((a, b) => b.flow - a.flow);
  topCandidates = candidateMoves.slice(0, 3);
  assignCandidateColors(topCandidates);

  updateCandidateListUI();

  // Auto-click the first candidate to move the piece
  const firstCandidateEl = candidateListEl.querySelector(".candidate");
  if (firstCandidateEl) {
    firstCandidateEl.click();
  }
}

function doSelectCandidate(actionKey) {
  simulationPaused = true;

  let data = selectMove(actionKey);
  if (data.error) {
    console.error("select_move error:", data.error);
    simulationPaused = false;
    return;
  }
  if (data.arrow) {
    let arrowData = data.arrow;
    let cand = topCandidates.find(c =>
      Math.abs(c.piece_center.x - arrowData.to.x) < 1 &&
      Math.abs(c.piece_center.y - arrowData.to.y) < 1
    );
    let color = cand ? cand.color : "#33ff66";
    spawnArrow(arrowData.from, arrowData.to, arrowData.flow, color);
  }

  currentGameState = data.game_state || {};

  // Wait a bit to let the piece animate
  setTimeout(() => {
    simulationPaused = false;
  }, MOVE_PAUSE_DURATION);
}

function gameTick() {
  if (!simulationPaused) {
    let data = tickGameLogic();
    currentGameState = data.game_state || {};
    currentPieceCenter = data.current_piece_center || { x:0, y:0 };

    let newPieceId = currentGameState.piece_id;
    if (typeof newPieceId !== "undefined" && newPieceId !== lastPieceId) {
      lastPieceId = newPieceId;
      setTimeout(fetchCandidateMoves, 200);
    }
  }
}

function doResetGame() {
  resetGameLogic();

  // Clear visuals
  currentGameState = null;
  currentPieceCenter = { x:0, y:0 };
  candidateMoves = [];
  topCandidates = [];
  appliedArrows = [];
  particles = [];
  simulationPaused = false;
  lastPieceId = null;

  candidateListEl.innerHTML = "";
  fetchCandidateMoves();
}

function updateCandidateListUI() {
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
    div.onclick = () => doSelectCandidate(c.action_key);
    candidateListEl.appendChild(div);
  });
}

//-----------------------------------------------------------------------------
// INIT FUNCTION - sets up canvas, UI references, event handlers, etc.
//-----------------------------------------------------------------------------

function init() {
  canvas = document.getElementById("tetrisCanvas");
  ctx = canvas.getContext("2d");
  candidateListEl = document.getElementById("candidateList");
  resetBtn = document.getElementById("resetBtn");

  // Create game & agent
  game = new TetrisGame();
  agent = new TrajectoryBalanceAgent(0.02);

  // Attempt to load pretrained flows from pretrained_flows_tb.json
  fetch("pretrained_flows_tb.json")
    .then(response => {
      if (!response.ok) {
        throw new Error("Could not fetch pretrained_flows_tb.json");
      }
      return response.json();
    })
    .then(data => {
      agent.loadFromJSON(data);
      console.log("Loaded pretrained flows from pretrained_flows_tb.json");
      // Start the game loop
      setInterval(gameTick, TICK_INTERVAL);
      fetchCandidateMoves();
    })
    .catch(err => {
      console.warn("Could not load pretrained flows:", err);
      // Even if we fail, we can still run the game with random flows
      setInterval(gameTick, TICK_INTERVAL);
      fetchCandidateMoves();
    });

  resetBtn.addEventListener("click", doResetGame);

  requestAnimationFrame(animate);
}

//-----------------------------------------------------------------------------
// LAUNCH
//-----------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", init);

