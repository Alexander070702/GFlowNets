#!/usr/bin/env python3
"""
pretrain.py

Final self-contained Tetris GFlowNet code that:
 - Pads piece orientations so rot=0..3 is always valid.
 - Uses a CNN to encode the 10x20 board.
 - Single-step distribution matching with heuristic reward.
 - Exports top-3 move JSON.

Usage: python3 pretrain.py
"""

import json, random, math, copy, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# ------------ Board Dimensions -----------
BOARD_COLS, BOARD_ROWS = 10, 20

# ------------ Tetromino Definitions -----------
TETROMINOES = {
    "I": [[1,1,1,1]],
    "O": [[1,1],
          [1,1]],
    "T": [[1,1,1],
          [0,1,0]],
    "S": [[0,1,1],
          [1,1,0]],
    "Z": [[1,1,0],
          [0,1,1]],
    "J": [[1,0,0],
          [1,1,1]],
    "L": [[0,0,1],
          [1,1,1]]
}
PIECES = list(TETROMINOES.keys())

def rotate_90(shape):
    """Rotate a 2D shape 90 degrees clockwise."""
    return [list(row) for row in zip(*shape[::-1])]

def all_orientations(piece_key):
    """
    Returns up to 4 distinct orientations for a tetromino.
    If fewer than 4 unique shapes, we pad with the last shape
    so that indexing [0..3] is always safe.
    """
    base = TETROMINOES[piece_key]
    shapes = []
    current = base
    for _ in range(4):
        if current not in shapes:
            shapes.append(current)
        current = rotate_90(current)
    while len(shapes) < 4:
        shapes.append(shapes[-1])  # pad
    return shapes

# ------------ Tetris Environment -----------
class TetrisGame:
    """
    Minimal Tetris environment for single-step piece drop.
    Spawns random piece, partially fills bottom row for variety.
    """
    def __init__(self):
        self.cols = BOARD_COLS
        self.rows = BOARD_ROWS
        self.reset()

    def reset(self):
        self.board = [[0]*self.cols for _ in range(self.rows)]
        self.game_over = False
        # partially fill bottom row
        row = [1]*self.cols
        empties = random.sample(range(self.cols), 3)
        for e in empties:
            row[e] = 0
        self.board[-1] = row
        self.spawn_piece()

    def spawn_piece(self):
        t_type = random.choice(PIECES)
        self.current_piece_type = t_type

    def collides(self, piece_type, rotation_idx, x, y):
        """Check collision out of board or with existing blocks."""
        shape = all_orientations(piece_type)[rotation_idx]
        for r, row in enumerate(shape):
            for c, val in enumerate(row):
                if val:
                    xx = x + c
                    yy = y + r
                    if xx<0 or xx>=self.cols or yy<0 or yy>=self.rows:
                        return True
                    if self.board[yy][xx] == 1:
                        return True
        return False

    def lock_piece(self, piece_type, rotation_idx, x, y, board_copy):
        """Lock piece onto board_copy at (x,y)."""
        shape = all_orientations(piece_type)[rotation_idx]
        for r, row in enumerate(shape):
            for c, val in enumerate(row):
                if val:
                    xx = x + c
                    yy = y + r
                    board_copy[yy][xx] = 1

    def clear_lines(self, board_copy):
        """Clear full lines and return the count."""
        newB = []
        cleared = 0
        for row in board_copy:
            if all(row):
                cleared += 1
            else:
                newB.append(row)
        while len(newB) < self.rows:
            newB.insert(0, [0]*self.cols)
        for r in range(self.rows):
            board_copy[r] = newB[r]
        return cleared

    def apply_move(self, piece_type, rotation_idx, col):
        """
        Drop piece from y=0 down until collision. Lock at y-1. 
        If y<0 => invalid => game_over.
        Returns (new_board, lines_cleared, final_y, game_over).
        """
        board_copy = copy.deepcopy(self.board)
        y = 0
        while True:
            if self.collides(piece_type, rotation_idx, col, y):
                y -= 1
                break
            y += 1
        if y < 0:
            # cannot place
            return board_copy, 0, -1, True
        # lock
        self.lock_piece(piece_type, rotation_idx, col, y, board_copy)
        lines_cleared = self.clear_lines(board_copy)
        return board_copy, lines_cleared, y, False

    def get_board_copy(self):
        return copy.deepcopy(self.board)

# ------------ Tetris Heuristic Reward -----------
def count_holes(board):
    rows = len(board)
    cols = len(board[0])
    holes = 0
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if board[r][c] == 1:
                block_found = True
            elif block_found and board[r][c] == 0:
                holes += 1
    return holes

def aggregate_height(board):
    rows = len(board)
    cols = len(board[0])
    total = 0
    for c in range(cols):
        col_h = 0
        for r in range(rows):
            if board[r][c] == 1:
                col_h = rows - r
                break
        total += col_h
    return total

def bumpiness(board):
    rows = len(board)
    cols = len(board[0])
    heights = []
    for c in range(cols):
        h = 0
        for r in range(rows):
            if board[r][c] == 1:
                h = rows - r
                break
        heights.append(h)
    bump = 0
    for c in range(cols-1):
        bump += abs(heights[c] - heights[c+1])
    return bump

def compute_heuristic_reward(board, lines_cleared):
    """
    Weighted sum then exponentiate:
     a=-0.51, b=0.76, c=-0.36, d=-0.18
     val = a*agg_h + b*lines_cleared + c*holes + d*bump
     R= exp(val)
    """
    a = -0.51
    b =  0.76
    c = -0.36
    d = -0.18

    agg_h = aggregate_height(board)
    holes = count_holes(board)
    bump = bumpiness(board)
    val = (a*agg_h) + (b*lines_cleared) + (c*holes) + (d*bump)
    return math.exp(val)

# ------------ CNN GFlowNet -----------
class CNNPolicyFlowNet(nn.Module):
    """
    Outputs 40 logits (4 rotations x 10 columns) + scalar flow (batch,).
    Input shape: (batch,1,20,10).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,padding=1)

        self.fc1 = nn.Linear(64*20*10,128)
        self.fc2 = nn.Linear(128,128)

        self.policy_head = nn.Linear(128,40)
        self.flow_head   = nn.Linear(128,1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        logits = self.policy_head(x)    # (batch,40)
        flow_log = self.flow_head(x)    # (batch,1)
        flow = torch.exp(flow_log).squeeze(-1)  # positive scalar
        return logits, flow

def board_to_tensor(board):
    """Convert (20x10) board to torch (1,20,10)."""
    arr = np.array(board, dtype=np.float32)  # shape(20,10)
    t = torch.from_numpy(arr).unsqueeze(0)   # (1,20,10)
    return t

# ------------ Single-Step GFlow Trainer -----------
class SingleStepGFlowTrainer:
    """
    For each random Tetris state, compute reward for each of 40 actions => target dist p(a).
    Cross-entropy + flow matching.  Replay buffer for mini-batch training.
    """
    def __init__(self, lr=1e-3, replay_size=50000, batch_size=64):
        self.net = CNNPolicyFlowNet()
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.replay = deque(maxlen=replay_size)
        self.batch_size = batch_size

        # define 4 rotations x 10 columns
        self.actions = [(r,c) for r in range(4) for c in range(BOARD_COLS)]

    def gather_experience(self, n_samples=10):
        for _ in range(n_samples):
            env = TetrisGame()
            if env.game_over:
                continue
            board = env.get_board_copy()
            piece_type = env.current_piece_type

            rewards = []
            valid_mask = []
            for (rot, col) in self.actions:
                new_board, lines, final_y, g_over = env.apply_move(piece_type, rot, col)
                if final_y<0 or g_over:
                    rewards.append(0.0)
                    valid_mask.append(False)
                else:
                    R = compute_heuristic_reward(new_board, lines)
                    rewards.append(R)
                    valid_mask.append(True)
            rewards = np.array(rewards, dtype=np.float32)
            valid_mask = np.array(valid_mask, dtype=bool)
            sumR = rewards.sum()
            if sumR<1e-9:
                # no valid moves
                continue
            dist = rewards / sumR
            self.replay.append((board, dist, sumR, valid_mask))

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        batch = random.sample(self.replay, self.batch_size)

        boards_list = []
        dists_list = []
        sumRs_list = []
        vmasks_list = []
        for (b, dist, sumR, vm) in batch:
            boards_list.append(board_to_tensor(b))
            dists_list.append(dist)
            sumRs_list.append(sumR)
            vmasks_list.append(vm)

        # stack boards
        boards_t = torch.stack(boards_list, dim=0)   # (batch,1,20,10)

        # convert distributions & masks to single arrays
        dists_arr = np.array(dists_list, dtype=np.float32)  # shape(batch,40)
        sumRs_arr = np.array(sumRs_list, dtype=np.float32)   # shape(batch,)
        vm_arr = np.array(vmasks_list, dtype=bool)           # shape(batch,40)

        dists_t = torch.from_numpy(dists_arr)       # (batch,40)
        sumRs_t = torch.from_numpy(sumRs_arr)        # (batch,)
        valid_masks_t = torch.from_numpy(vm_arr)     # (batch,40)

        self.net.train()
        logits, flow = self.net(boards_t)  # (batch,40), (batch,)

        # mask invalid
        logits_masked = logits.clone()
        logits_masked[~valid_masks_t] = -1e9

        log_probs = torch.log_softmax(logits_masked, dim=-1)
        ce_loss = -(dists_t * log_probs).sum(dim=-1).mean()

        fm_loss = nn.functional.mse_loss(flow, sumRs_t)

        loss = ce_loss + fm_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def predict_action_distribution(self, board):
        """Return (logits, flow_val) for single board."""
        self.net.eval()
        with torch.no_grad():
            bt = board_to_tensor(board).unsqueeze(0)  # (1,1,20,10)
            logits, flow = self.net(bt)
            return logits[0], flow[0].item()

    def save_model(self, path="cnn_gfn_tetris.pt"):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path="cnn_gfn_tetris2.pt"):
        self.net.load_state_dict(torch.load(path))

# ------------ Top-3 Moves JSON -----------
def get_top_moves_json(board, piece_type, trainer, top_k=3):
    """
    Evaluate policy => mask invalid => pick top_k => output JSON with
    rotation, column, flow, probability.
    """
    logits, flow_val = trainer.predict_action_distribution(board)
    env = TetrisGame()
    env.board = copy.deepcopy(board)
    env.current_piece_type = piece_type

    valid_mask = []
    for (rot, col) in trainer.actions:
        new_board, lines, final_y, g_over = env.apply_move(piece_type, rot, col)
        if final_y<0 or g_over:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
    valid_mask = np.array(valid_mask,dtype=bool)

    logits_masked = logits.clone()
    logits_masked[~valid_mask] = -1e9
    probs = torch.softmax(logits_masked, dim=-1).numpy()

    top_indices = np.argsort(-probs)[:top_k]
    candidates = []
    for idx in top_indices:
        (rot, col) = trainer.actions[idx]
        p = probs[idx]
        local_flow = flow_val*p
        candidates.append({
            "rotation": int(rot),
            "column": int(col),
            "flow": round(float(local_flow),4),
            "probability": round(float(p),4)
        })

    return {
        "piece": piece_type,
        "candidates": candidates
    }

# ------------ Training Function -----------
def train_superhuman_tetris(n_iterations=1000):
    trainer = SingleStepGFlowTrainer(lr=1e-3, replay_size=50000, batch_size=64)
    losses = []
    print_interval = 100

    for it in range(n_iterations):
        # gather random states
        trainer.gather_experience(n_samples=10)
        # do multiple train steps
        ep_loss=0.0
        for _ in range(5):
            l = trainer.train_step()
            ep_loss+=l
        ep_loss/=5.0
        losses.append(ep_loss)

        if (it+1)%print_interval==0:
            print(f"[{it+1}/{n_iterations}] loss={ep_loss:.4f}")

    # save
    trainer.save_model("cnn_gfn_tetris.pt")
    # plot
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Single-Step GFlowNet Tetris Training")
    plt.savefig("cnn_gfn_tetris_loss.png")
    plt.close()

    return trainer

# ------------ Main -----------
if __name__=="__main__":
    trainer = train_superhuman_tetris(n_iterations=100)
    # show top-3 moves for a random board
    game = TetrisGame()
    game.reset()
    if not game.game_over:
        board = game.get_board_copy()
        ptype = game.current_piece_type
        json_moves = get_top_moves_json(board, ptype, trainer, top_k=3)
        print("Top 3 moves JSON for piece:", ptype)
        print(json.dumps(json_moves, indent=2))
    print("Done. Model saved to cnn_gfn_tetris.pt")
