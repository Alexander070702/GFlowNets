#!/usr/bin/env python3
"""
final_train.py

Trains a Tetris GFlowNet via a neural network policy using:
 - CNN board encoder + piece embeddings
 - Dueling Trajectory‑Balance GFlowNet (learnable logZ)
 - Prioritized experience replay
 - Temperature annealing, weight decay, scheduler, gradient clipping
 - Reward shaping: line clears, survival bonus, Tetris bonus, perfect‑clear,
   hole/height/bumpiness penalties (with adjustable height penalty)

Every 500 episodes (and at the end) it exports:
  - pretrained_flows_nn.json    (head weights + logZ)
  - encoder_weights.json        (conv1/conv2 kernels & biases, embed matrix)

Run:
    python3 final_train.py
"""

import os
import json
import math
import copy
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# ===== Device setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== Hyperparameters =====
BOARD_COLS, BOARD_ROWS = 10, 20
FLOW_JSON_FILE    = "pretrained_flows_nn.json"
ENCODER_JSON_FILE = "encoder_weights.json"
TOTAL_EPISODES    = 50000
BATCH_SIZE        = 64
LEARNING_RATE     = 1e-4
EPSILON_START     = 0.05  # reduced exploration for effective learning
HEIGHT_PENALTY    = 1.0    # height penalty coefficient
HOLE_PENALTY      = 2.0
BUMP_PENALTY      = 1.0
ACTION_DIM        = 2

# ===== Tetris Environment =====
TETROMINOES = {
    "I": [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
    "O": [[1,1],[1,1]],
    "T": [[0,1,0],[1,1,1],[0,0,0]],
    "S": [[0,1,1],[1,1,0],[0,0,0]],
    "Z": [[1,1,0],[0,1,1],[0,0,0]],
    "J": [[1,0,0],[1,1,1],[0,0,0]],
    "L": [[0,0,1],[1,1,1],[0,0,0]],
}
PIECES    = list(TETROMINOES.keys())
PIECE_IDS = {k:i for i,k in enumerate(PIECES)}

def rotate(matrix):
    return [list(row) for row in zip(*matrix[::-1])]

class TetrisGame:
    def __init__(self):
        self.cols, self.rows = BOARD_COLS, BOARD_ROWS
        self.reset_game()

    def reset_game(self):
        self.board = [[0]*self.cols for _ in range(self.rows)]
        # seed holes on bottom row
        bottom = [1]*self.cols
        for e in random.sample(range(self.cols), 3):
            bottom[e] = 0
        self.board[-1] = bottom
        self.score = 0.0
        self.game_over = False
        self.spawn_piece()

    def spawn_piece(self):
        t = random.choice(PIECES)
        self.current_piece = {
            "type": t,
            "shape": copy.deepcopy(TETROMINOES[t]),
            "x": self.cols//2,
            "y": 0
        }
        if self.collides(self.current_piece):
            self.game_over = True

    def collides(self, piece):
        for r,row in enumerate(piece["shape"]):
            for c,val in enumerate(row):
                if not val: continue
                x,y = piece["x"]+c, piece["y"]+r
                if x<0 or x>=self.cols or y<0 or y>=self.rows: return True
                if y>=0 and self.board[y][x]: return True
        return False

    def clear_lines(self):
        newb = [r for r in self.board if not all(r)]
        lines = self.rows - len(newb)
        for _ in range(lines):
            newb.insert(0, [0]*self.cols)
        self.board = newb
        return lines

    def get_top_height(self):
        for r,row in enumerate(self.board):
            if any(row): return self.rows - r
        return 0

    def count_holes(self):
        holes = 0
        for c in range(self.cols):
            seen = False
            for r in range(self.rows):
                if self.board[r][c]:
                    seen = True
                elif seen:
                    holes += 1
        return holes

    def count_bumpiness(self):
        heights = [
            next((self.rows-r for r in range(self.rows) if self.board[r][c]), 0)
            for c in range(self.cols)
        ]
        return sum(abs(heights[i] - heights[i+1]) for i in range(self.cols-1))

    def lock_piece(self, piece):
        for r,row in enumerate(piece["shape"]):
            for c,val in enumerate(row):
                if val:
                    x,y = piece["x"]+c, piece["y"]+r
                    if 0<=x<self.cols and 0<=y<self.rows:
                        self.board[y][x] = 1
        return self.clear_lines()

    def apply_move(self, piece):
        lines = self.lock_piece(piece)

        # Base reward for lines cleared
        base = {1: 300, 2: 900, 3: 1800, 4: 3000}.get(lines, 0)
        reward = base

        # Bonus for Tetris
        if lines == 4:
            reward += 800  # stronger reward for Tetris

        # Perfect clear bonus
        if all(cell == 0 for row in self.board for cell in row):
            reward += 1500

        # Survival bonus (constant per step)
        reward += 1

        # Exploration encouragement: small random bonus
        reward += random.uniform(0, 5)

        # Penalties
        height_penalty = (self.get_top_height() ** 1.5) * HEIGHT_PENALTY
        hole_penalty = self.count_holes() * HOLE_PENALTY
        bump_penalty = self.count_bumpiness() * BUMP_PENALTY
        penalty = hole_penalty + height_penalty + bump_penalty

        self.score += reward - penalty

        # Spawn new piece
        self.spawn_piece()
        if self.collides(self.current_piece):
            self.game_over = True

    def get_moves(self):
        if self.game_over: return []
        cands = []
        t = self.current_piece["type"]
        rots = 1 if t=="O" else 4
        for rot in range(rots):
            shape = copy.deepcopy(TETROMINOES[t])
            for _ in range(rot): shape = rotate(shape)
            w = len(shape[0])
            for x in range(self.cols-w+1):
                test = {"type":t, "shape":copy.deepcopy(shape), "x":x, "y":0}
                if self.collides(test): continue
                while not self.collides(test): test["y"] += 1
                test["y"] -= 1
                if test["y"] < 0: continue
                cands.append((f"r{rot}_x{x}", test))
        return cands

    def is_over(self): return self.game_over
    def get_final_reward(self): return self.score

# ===== Model definitions =====
class BoardEncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
    def forward(self, x):
        return self.conv(x)

class DuelFlowNetHead(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.shared     = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.state_flow = nn.Linear(hidden, 1)
        self.advantage  = nn.Linear(hidden + ACTION_DIM, 1)
    def forward(self, s_feat, a_feat):
        h   = self.shared(s_feat)
        z   = self.state_flow(h)
        adv = self.advantage(torch.cat([h, a_feat], dim=-1))
        return z + (adv - adv.mean(dim=0, keepdim=True))

class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
        self.buf, self.cap, self.alpha = [], capacity, alpha
    def push(self, traj, r, p):
        if len(self.buf) >= self.cap: self.buf.pop(0)
        self.buf.append({"traj":traj, "r":r, "p":p})
    def sample(self, k):
        ps = np.array([e["p"]**self.alpha for e in self.buf], dtype=float)
        ps /= ps.sum()
        idxs = np.random.choice(len(self.buf), k, p=ps)
        return [self.buf[i] for i in idxs]

class GFlowNetAgent:
    def __init__(self, total_episodes):
        self.encoder = BoardEncoderCNN().to(device)
        self.embed   = nn.Embedding(len(PIECES), 16).to(device)
        dummy = torch.zeros(1,1,BOARD_ROWS,BOARD_COLS, device=device)
        conv_dim = self.encoder(dummy).shape[-1]
        state_dim = conv_dim + 16
        self.head   = DuelFlowNetHead(state_dim).to(device)
        self.logZ   = nn.Parameter(torch.tensor(0.0, device=device))
        self.opt    = optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.embed.parameters())   +
            list(self.head.parameters())    +
            [self.logZ],
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )
        self.sched  = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=total_episodes)
        self.buffer      = PrioritizedReplayBuffer()
        self.batch_size  = BATCH_SIZE
        self.epsilon     = EPSILON_START
        self.temperature = 1.0

    def propose_action(self, board, piece, candidates):
        if not candidates: return None
        # ε-greedy sampling
        if random.random() < self.epsilon:
            return random.choice(candidates)
        with torch.no_grad():
            b_t   = torch.tensor(board, dtype=torch.float32, device=device).view(1,1,BOARD_ROWS,BOARD_COLS)
            cnn_f = self.encoder(b_t).squeeze(0)
            pid   = torch.tensor([PIECE_IDS[piece["type"]]], device=device)
            emb_f = self.embed(pid).squeeze(0)
            s_feat = torch.cat([cnn_f, emb_f], dim=-1)
            a_feats = torch.stack([
                torch.tensor([
                    int(k.split("_")[0][1:]) / 3.0,
                    int(k.split("_")[1][1:]) / (BOARD_COLS - 1)
                ], dtype=torch.float32, device=device)
                for k,_ in candidates
            ])
            s_feats = s_feat.unsqueeze(0).repeat(len(candidates), 1)
            logf    = self.head(s_feats, a_feats).squeeze(-1)
            logf    = torch.clamp(logf, -20, 20)
            probs   = torch.softmax(logf / self.temperature, dim=0)
            idx     = torch.multinomial(probs, 1).item()
        return candidates[idx]

    def add_trajectory(self, traj, final_reward):
        r = max(final_reward, 1e-6)
        p = abs(math.log(r))
        self.buffer.push(traj, final_reward, p)

    def update(self):
        if len(self.buffer.buf) < self.batch_size: return None
        batch = self.buffer.sample(self.batch_size)
        self.opt.zero_grad()
        losses = []
        for entry in batch:
            traj, R = entry["traj"], max(entry["r"], 1e-6)
            logR    = torch.log(torch.tensor(R, device=device))
            s_list, a_list, offsets = [], [], []
            start = 0
            for board, piece, chosen_key, all_keys in traj:
                b_t   = torch.tensor(board, dtype=torch.float32, device=device).view(1,1,BOARD_ROWS,BOARD_COLS)
                cnn_f = self.encoder(b_t).squeeze(0)
                pid   = torch.tensor([PIECE_IDS[piece["type"]]], device=device)
                emb_f = self.embed(pid).squeeze(0)
                s_emb = torch.cat([cnn_f, emb_f], dim=-1)
                for k in all_keys:
                    s_list.append(s_emb)
                    a_list.append(torch.tensor([
                        int(k.split("_")[0][1:]) / 3.0,
                        int(k.split("_")[1][1:]) / (BOARD_COLS - 1)
                    ], dtype=torch.float32, device=device))
                L  = len(all_keys)
                ci = all_keys.index(chosen_key)
                offsets.append((start, start+L, ci))
                start += L
            s_feats = torch.stack(s_list)
            a_feats = torch.stack(a_list)
            logf = self.head(s_feats, a_feats).squeeze(-1)
            logf = torch.clamp(logf, -20, 20)
            sum_logp = 0.0
            for s,e,ci in offsets:
                seg = logf[s:e]
                mx  = seg.max()
                lse = mx + torch.log((seg - mx).exp().sum())
                sum_logp += seg[ci] - lse
            losses.append((sum_logp + self.logZ - logR)**2)
        loss = torch.stack(losses).mean()
        loss.backward()
        clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.head.parameters()), 1.0
        )
        self.opt.step()
        self.sched.step()
        return loss.item()

    def export_json(self):
        out = {"logZ": self.logZ.item(), "weights": {}}
        for name,param in self.head.named_parameters():
            out["weights"][name] = param.detach().cpu().tolist()
        with open(FLOW_JSON_FILE, "w") as f:
            json.dump(out, f, indent=2)
        print(f"👉 Wrote {FLOW_JSON_FILE}")

# ===== Training Loop =====
def train_tetris_nn():
    agent = GFlowNetAgent(TOTAL_EPISODES)
    rewards, best = [], -float("inf")

    for ep in range(TOTAL_EPISODES):
        game, traj = TetrisGame(), []
        while not game.is_over():
            cands = game.get_moves()
            if not cands: break
            chosen = agent.propose_action(game.board, game.current_piece, cands)
            if chosen is None: break
            key, placement = chosen
            traj.append((
                copy.deepcopy(game.board),
                copy.deepcopy(game.current_piece),
                key,
                [a[0] for a in cands]
            ))
            game.apply_move(placement)
        R = game.get_final_reward()
        best = max(best, R)
        rewards.append(R)
        agent.add_trajectory(traj, R)
        agent.temperature = max(0.1, 1.0 - ep / TOTAL_EPISODES)
        for _ in range(5): agent.update()
        if ep and ep % 100 == 0:
            avg100 = np.mean(rewards[-100:])
            print(f"[{ep:5d}] avg100={avg100:.1f}, best={best:.1f}")
        if (ep+1) % 500 == 0:
            agent.export_json()
            conv1 = agent.encoder.conv[0]
            conv2 = agent.encoder.conv[2]
            emb   = agent.embed
            enc_wts = {
                "conv1": {"kernel": conv1.weight.detach().cpu().numpy().tolist(),
                          "bias":   conv1.bias.detach().cpu().numpy().tolist()},
                "conv2": {"kernel": conv2.weight.detach().cpu().numpy().tolist(),
                          "bias":   conv2.bias.detach().cpu().numpy().tolist()},
                "embed": emb.weight.detach().cpu().numpy().tolist()
            }
            with open(ENCODER_JSON_FILE, "w") as f:
                json.dump(enc_wts, f, indent=2)
            print(f"👉 Wrote {ENCODER_JSON_FILE}")
    agent.export_json()
    conv1,conv2,emb = agent.encoder.conv[0], agent.encoder.conv[2], agent.embed
    enc_wts = {
        "conv1": {"kernel": conv1.weight.detach().cpu().numpy().tolist(),
                  "bias":   conv1.bias.detach().cpu().numpy().tolist()},
        "conv2": {"kernel": conv2.weight.detach().cpu().numpy().tolist(),
                  "bias":   conv2.bias.detach().cpu().numpy().tolist()},
        "embed": emb.weight.detach().cpu().numpy().tolist()
    }
    with open(ENCODER_JSON_FILE, "w") as f:
        json.dump(enc_wts, f, indent=2)
    print(f"👉 Wrote {ENCODER_JSON_FILE} (final)")
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("GFlowNet Tetris Training")
    plt.savefig("training_curve.png")
    print("👉 Saved training_curve.png")

if __name__ == "__main__":
    train_tetris_nn()
