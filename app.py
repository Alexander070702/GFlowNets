#!/usr/bin/env python3
from flask import Flask, jsonify, request, render_template
import random, json, copy, os, math

app = Flask(__name__)

COLS = 10
ROWS = 20
CELL_SIZE = 30

TETROMINOES = {
    'I': [
        [0,0,0,0],
        [1,1,1,1],
        [0,0,0,0],
        [0,0,0,0]
    ],
    'O': [
        [1,1],
        [1,1]
    ],
    'T': [
        [0,1,0],
        [1,1,1],
        [0,0,0]
    ],
    'S': [
        [0,1,1],
        [1,1,0],
        [0,0,0]
    ],
    'Z': [
        [1,1,0],
        [0,1,1],
        [0,0,0]
    ],
    'J': [
        [1,0,0],
        [1,1,1],
        [0,0,0]
    ],
    'L': [
        [0,0,1],
        [1,1,1],
        [0,0,0]
    ]
}

def deep_copy(matrix):
    return copy.deepcopy(matrix)

def rotate_matrix(matrix):
    """Rotate matrix 90 degrees clockwise."""
    return [list(reversed(col)) for col in zip(*matrix)]

class TetrisGame:
    """
    GFlowNet Tetris. We maintain a piece_id so we can detect when a new piece spawns.
    """
    def __init__(self, cols=COLS, rows=ROWS):
        self.cols = cols
        self.rows = rows
        self.piece_id = 0  # increments every time we spawn a new piece
        self.reset_game()

    def reset_game(self):
        self.board = [[0]*self.cols for _ in range(self.rows)]
        self.score = 0
        self.game_over = False
        self.piece_id = 0
        self.current_piece = self.spawn_piece()
        self.target_piece = None
        self.cached_moves = None
        self._cached_state_key = None

    def spawn_piece(self):
        self.piece_id += 1  # each new piece increments piece_id
        t_type = random.choice(list(TETROMINOES.keys()))
        shape = deep_copy(TETROMINOES[t_type])
        piece = {
            'type': t_type,
            'shape': shape,
            'x': (self.cols - len(shape[0])) // 2,
            'y': 0
        }
        if self.collides(piece):
            self.game_over = True
        return piece

    def collides(self, piece):
        shape = piece['shape']
        for r in range(len(shape)):
            for c in range(len(shape[r])):
                if shape[r][c]:
                    x = piece['x'] + c
                    y = piece['y'] + r
                    if x < 0 or x >= self.cols or y >= self.rows:
                        return True
                    if y >= 0 and self.board[y][x]:
                        return True
        return False

    def clear_lines(self):
        new_board = [row for row in self.board if not all(cell == 1 for cell in row)]
        cleared = self.rows - len(new_board)
        for _ in range(cleared):
            new_board.insert(0, [0]*self.cols)
        self.board = new_board
        self.score += cleared
        return cleared

    def lock_piece(self):
        p = self.current_piece
        for r in range(len(p['shape'])):
            for c in range(len(p['shape'][r])):
                if p['shape'][r][c]:
                    x = p['x'] + c
                    y = p['y'] + r
                    if 0 <= x < self.cols and 0 <= y < self.rows:
                        self.board[y][x] = 1
        self.clear_lines()
        self.current_piece = self.spawn_piece()
        self.target_piece = None
        self.cached_moves = None
        self._cached_state_key = None

    def lock_target(self):
        if not self.target_piece:
            return
        p = self.target_piece
        for r in range(len(p['shape'])):
            for c in range(len(p['shape'][r])):
                if p['shape'][r][c]:
                    x = p['x'] + c
                    y = p['y'] + r
                    if 0 <= x < self.cols and 0 <= y < self.rows:
                        self.board[y][x] = 1
        self.clear_lines()
        self.current_piece = self.spawn_piece()
        self.target_piece = None
        self.cached_moves = None
        self._cached_state_key = None

    def get_piece_center(self, piece=None):
        if piece is None:
            piece = self.current_piece
        if not piece or not piece.get('shape'):
            return {'x':0, 'y':0}
        h = len(piece['shape'])
        w = len(piece['shape'][0])
        return {
            'x': (piece['x'] + w/2)*CELL_SIZE,
            'y': (piece['y'] + h/2)*CELL_SIZE
        }

    def get_state_key(self):
        """
        Include board, piece type, position, AND the current piece shape.
        This ensures that any rotation (change in shape) updates the state key.
        """
        return json.dumps({
            'board': self.board,
            'piece': {
                'type': self.current_piece['type'],
                'shape': self.current_piece['shape'],
                'x': self.current_piece['x'],
                'y': self.current_piece['y']
            }
        })

    def get_terminal_moves(self):
        if self.game_over:
            return []
        current_state_key = self.get_state_key()
        if self.cached_moves is not None and self._cached_state_key == current_state_key:
            return self.cached_moves

        orig = self.current_piece
        base_shape = TETROMINOES[orig['type']]
        candidates = []
        rotations = [0] if orig['type'] == 'O' else range(4)
        for rot in rotations:
            shape = deep_copy(base_shape)
            for _ in range(rot):
                shape = rotate_matrix(shape)
            h = len(shape)
            w = len(shape[0])
            for x in range(self.cols - w + 1):
                test = {
                    'type': orig['type'],
                    'shape': deep_copy(shape),
                    'x': x,
                    'y': 0
                }
                if self.collides(test):
                    continue
                y = 0
                while not self.collides({**test, 'y': y}) and y < self.rows:
                    y += 1
                test['y'] = y - 1
                if test['y'] < 0:
                    continue
                center = self.get_piece_center(test)
                action_key = f"r{rot}_x{x}"
                candidates.append({
                    'action_key': action_key,
                    'piece': test,
                    'piece_center': center
                })
        self.cached_moves = candidates
        self._cached_state_key = current_state_key
        return candidates

    def tick(self):
        if self.game_over:
            return
        if self.target_piece:
            # Update shape immediately to reflect rotation changes
            self.current_piece['shape'] = deep_copy(self.target_piece['shape'])
            # Invalidate cached terminal moves whenever the piece changes
            self.cached_moves = None
            self._cached_state_key = None

            px = self.current_piece['x']
            py = self.current_piece['y']
            tx = self.target_piece['x']
            ty = self.target_piece['y']
            if px < tx:
                self.current_piece['x'] += 1
            elif px > tx:
                self.current_piece['x'] -= 1
            if py < ty:
                self.current_piece['y'] += 1
            elif py > ty:
                self.current_piece['y'] -= 1
            if self.current_piece['x'] == tx and self.current_piece['y'] == ty:
                self.lock_target()
        else:
            next_p = {
                'type': self.current_piece['type'],
                'shape': self.current_piece['shape'],
                'x': self.current_piece['x'],
                'y': self.current_piece['y'] + 1
            }
            if not self.collides(next_p):
                self.current_piece['y'] += 1
            else:
                self.lock_piece()

    def is_over(self):
        return self.game_over

    def get_final_reward(self):
        if self.game_over:
            return self.score * 10 - 10
        else:
            return self.score * 10

# ------------------ Trajectory-Balance GFlowNet Agent ------------------ #
class TrajectoryBalanceAgent:
    def __init__(self, lr=0.01):
        self.log_flows = {}
        self.logZ = 0.0
        self.lr = lr

    def _ensure_action_exists(self, state_key, action_key):
        if state_key not in self.log_flows:
            self.log_flows[state_key] = {}
        if action_key not in self.log_flows[state_key]:
            self.log_flows[state_key][action_key] = math.log(0.5 + random.random())

    def sample_action(self, state_key, candidates):
        for c in candidates:
            self._ensure_action_exists(state_key, c['action_key'])
        log_values = [self.log_flows[state_key][c['action_key']] for c in candidates]
        max_log = max(log_values)
        exps = [math.exp(lv - max_log) for lv in log_values]
        sum_exps = sum(exps)
        probs = [e/sum_exps for e in exps]
        r = random.random()
        cum = 0.0
        idx = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                idx = i
                break
        return candidates[idx], probs[idx]

    def get_log_p_action(self, state_key, action_key):
        self._ensure_action_exists(state_key, action_key)
        all_logs = list(self.log_flows[state_key].values())
        max_val = max(all_logs)
        denom = math.log(sum(math.exp(x - max_val) for x in all_logs)) + max_val
        numerator = self.log_flows[state_key][action_key]
        return numerator - denom

    def update_trajectory(self, trajectory, final_reward):
        if final_reward <= 0:
            final_reward = 0.01
        logR = math.log(final_reward)
        sum_logp = sum(self.get_log_p_action(s, a) for (s, a) in trajectory)
        target = logR - self.logZ
        diff = sum_logp - target
        self.logZ += self.lr * diff
        for (s, a) in trajectory:
            self.log_flows[s][a] -= self.lr * diff

    def save(self, fname):
        data = {
            "log_flows": {
                s: {a: lf for a, lf in adict.items()}
                for s, adict in self.log_flows.items()
            },
            "logZ": self.logZ
        }
        with open(fname, "w") as f:
            json.dump(data, f)

    def load(self, fname):
        try:
            with open(fname, "r") as f:
                data = json.load(f)
            self.log_flows = {
                s: {a: lf for a, lf in adict.items()}
                for s, adict in data["log_flows"].items()
            }
            self.logZ = data["logZ"]
        except Exception as e:
            print("Error loading agent:", e)

# ------------------ END AGENT DEFINITION ------------------ #

game = TetrisGame()
agent = TrajectoryBalanceAgent(lr=0.02)
trajectory = []

if os.path.exists("pretrained_flows_tb.json"):
    agent.load("pretrained_flows_tb.json")
    print("TrajectoryBalance flows loaded from pretrained_flows_tb.json.")

from flask import render_template

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api/terminal_moves', methods=['GET'])
def api_terminal_moves():
    """
    If your front-end calls this, it returns the candidate moves for the current piece.
    """
    if game.is_over():
        return jsonify({
            'current_piece_center': game.get_piece_center(),
            'terminal_moves': [],
            'game_state': {
                'board': game.board,
                'current_piece': game.current_piece,
                'score': game.score,
                'game_over': game.game_over
            }
        })
    state_key = game.get_state_key()
    cands = game.get_terminal_moves()
    result = []
    sum_exp = 0.0
    for c in cands:
        agent._ensure_action_exists(state_key, c['action_key'])
        val = math.exp(agent.log_flows[state_key][c['action_key']])
        sum_exp += val
    for c in cands:
        flow_val = math.exp(agent.log_flows[state_key][c['action_key']])
        prob = flow_val / sum_exp if sum_exp > 0 else 1.0 / len(cands)
        c['flow'] = flow_val
        c['probability'] = prob
        result.append(c)
    return jsonify({
        'current_piece_center': game.get_piece_center(),
        'terminal_moves': result,
        'game_state': {
            'board': game.board,
            'current_piece': game.current_piece,
            'score': game.score,
            'game_over': game.game_over
        }
    })

@app.route('/api/select_move', methods=['POST'])
def api_select_move():
    global trajectory
    data = request.get_json()
    chosen_key = data.get('action_key')
    if game.is_over():
        return jsonify({'error': 'Game Over'}), 400

    cands = game.get_terminal_moves()
    if not cands:
        return jsonify({'error': 'No moves'}), 400

    state_key = game.get_state_key()
    selected_action = None
    if chosen_key:
        selected_action = next((x for x in cands if x['action_key'] == chosen_key), None)
    if not selected_action:
        selected_action, _ = agent.sample_action(state_key, cands)

    # Record the chosen (state, action) for TB updates later
    trajectory.append((state_key, selected_action['action_key']))

    # Set the piece as the target piece, so tick() will animate it
    game.target_piece = copy.deepcopy(selected_action['piece'])

    arrow_info = {
        'from': game.get_piece_center(game.current_piece),
        'to': selected_action['piece_center'],
        'flow': 0.0,
        'probability': 0.0
    }
    return jsonify({
        'action_key': selected_action['action_key'],
        'arrow': arrow_info,
        'game_state': {
            'board': game.board,
            'current_piece': game.current_piece,
            'score': game.score,
            'game_over': game.game_over,
            'piece_id': game.piece_id
        }
    })

@app.route('/api/tick', methods=['POST'])
def api_tick():
    """
    This is typically called on a game loop or setInterval to animate the piece downward.
    If a new piece spawns, we recalc the terminal moves and return them so the front-end
    always has the latest candidate moves.
    """
    global trajectory
    old_game_over = game.is_over()
    old_piece_id = game.piece_id

    game.tick()  # animate or drop piece

    new_game_over = game.is_over()
    new_piece_id = game.piece_id

    # If the game ended just now, do TB update and reset if you wish
    if new_game_over and not old_game_over:
        final_reward = game.get_final_reward()
        agent.update_trajectory(trajectory, final_reward)
        trajectory.clear()
        game.reset_game()

    # If a new piece was spawned, recalc the new terminal moves right away
    terminal_moves = []
    if new_piece_id != old_piece_id and not game.is_over():
        # The old piece locked, new piece spawned => recalc moves
        state_key = game.get_state_key()
        cands = game.get_terminal_moves()
        sum_exp = 0.0
        for c in cands:
            agent._ensure_action_exists(state_key, c['action_key'])
            val = math.exp(agent.log_flows[state_key][c['action_key']])
            sum_exp += val
        for c in cands:
            flow_val = math.exp(agent.log_flows[state_key][c['action_key']])
            prob = flow_val / sum_exp if sum_exp > 0 else 1.0 / len(cands)
            c['flow'] = flow_val
            c['probability'] = prob
            terminal_moves.append(c)

    return jsonify({
        'game_state': {
            'board': game.board,
            'current_piece': game.current_piece,
            'score': game.score,
            'game_over': game.game_over,
            'piece_id': game.piece_id
        },
        'current_piece_center': game.get_piece_center(),
        # Return the new terminal moves if a piece was spawned
        'terminal_moves': terminal_moves
    })

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global trajectory
    game.reset_game()
    trajectory.clear()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True)
