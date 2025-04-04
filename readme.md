# GFlowNet Tetris Demo

This project demonstrates a **Generative Flow Network (GFlowNet)** applied to a simplified Tetris game. The code includes two main scripts:

1. **`app.py`** – Runs a Flask server that hosts the Tetris game in a browser, exposes REST endpoints for selecting moves, and animates the game.
2. **`pretrain.py`** – Offline pretraining script that simulates many Tetris games, learning GFlowNet “flows” (log probabilities) via the **Trajectory Balance** (TB) method.

You’ll also find:

- **`templates/index.html`** – The main HTML file that loads the Tetris UI in your browser.
- **`static/`** – JavaScript files for interactive visualizations, plus CSS for styling.

---

## 1. How GFlowNets Work Here

A **Generative Flow Network (GFlowNet)** is a framework for sampling states (in this case, Tetris placements) in proportion to a reward function. Rather than focusing on one single “optimal” sequence of moves, a GFlowNet aims to learn a policy that **samples multiple good solutions** at frequencies matching their reward.

### Trajectory Balance (TB)

We use the **Trajectory Balance** objective, which enforces:

\[
\sum_{(s,a)\in\tau} \log \pi_{\theta}(a \mid s) \;\approx\; \log R(\tau) \;-\; \log Z,
\]

where:
- \( \tau \) is a trajectory from start to game over,
- \( R(\tau) \) is the final reward (number of lines cleared minus a penalty if the board fills up),
- \( Z \) is a learnable normalization constant,
- \( \pi_{\theta}(a \mid s) \) is the GFlowNet policy derived from the learned \(\log F\) flows.

Over many training episodes, **TB** aligns the product of action probabilities along each trajectory with its reward, effectively learning to sample high-reward paths more frequently.

---

## 2. `app.py` in Detail

### TetrisGame Class

- **Board Representation**: A 2D list (`self.board`) holding 0 (empty) or 1 (occupied).
- **Piece Spawning**: Each new piece increments a `piece_id`, so we can detect when a new piece arrives.
- **Collision Checks**: Ensure pieces don’t overlap the board edges or existing blocks.
- **Locking Pieces**: When a piece can’t fall further, it’s locked into the board, full lines are cleared, and the next piece spawns.
- **State Key**: `get_state_key()` returns a JSON representation of the board plus the current piece’s position, shape, and type. GFlowNet uses this key to identify states.

### TrajectoryBalanceAgent Class

- Stores a dictionary of `log_flows[state][action]` and a global `logZ`.
- **Sampling**: From a given state, it uses a softmax of `logF(state->action)` to pick an action.
- **TB Update**: At game over, it adjusts \(\log F\) and \(\log Z\) so that the product of probabilities over the chosen trajectory matches \(\frac{R(\tau)}{Z}\).

### Flask Endpoints

1. **`/`**  
   Serves `index.html`, the main Tetris UI.

2. **`/api/terminal_moves`**  
   Returns all final placements (“terminal” moves) for the current piece, each with an unnormalized flow and probability.

3. **`/api/select_move`**  
   Called by the front-end to pick one of the terminal moves. The server logs the `(state, action)` pair for TB updates, sets `target_piece`, and returns basic info for front-end animations.

4. **`/api/tick`**  
   Called on a loop or interval to drop or animate the current piece. If a piece is locked and a new one spawns, we recalc terminal moves. If the game ends, we apply the TB update and reset if desired.

5. **`/api/reset`**  
   Resets the game and clears any in-progress trajectory data.

### Front-End Integration

- The front-end (in `static/`) calls these endpoints to:
  - Retrieve terminal moves and their GFlowNet flows.
  - Animate piece movement (via `/api/tick`).
  - Show “Top Candidate Moves” with their flows/probabilities.
- `templates/index.html` hosts the main HTML structure and includes references to the JS/CSS in `static/`.

---

## 3. `pretrain.py` in Detail

This script does offline training without the browser UI:

1. **Simulate Episodes**  
   Creates a smaller Tetris board (e.g., 6×10), repeatedly spawns pieces, and picks final placements via the GFlowNet policy.

2. **Final Reward**  
   Once the board is full or no moves remain, we compute the final reward (`lines_cleared * 10 - 10` if game over) and call `agent.update_trajectory(...)`.

3. **Checkpointing**  
   Every `checkpoint_interval` episodes, the script saves `log_flows` and `logZ` to a JSON file (e.g., `pretrained_flows_tb.json`). This file can be loaded by `app.py` to start with a partially trained policy.

---

## 4. Templates & Static Files

- **`templates/index.html`**  
  Contains the HTML layout, including elements for displaying the Tetris board and any flow visualizations.

- **`static/`**  
  - **JavaScript**: Interactive demos, front-end logic for calling `/api/tick`, `/api/select_move`, etc.
  - **CSS**: Styles the Tetris grid, flow animations, or other UI elements.

---

## 5. How to Run

1. **Install Dependencies**  
   - Python 3, plus `Flask`.

2. **Optional: Pretrain**  
   ```bash
   python pretrain.py --episodes 20000 --checkpoint_file pretrained_flows_tb.json
