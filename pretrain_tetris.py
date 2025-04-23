#!/usr/bin/env python3
"""
pretrain.py

Trains a Tetris GFlowNet via a neural network policy with partial line rewards, 
and negative penalties for holes, height, bumpiness to encourage "clean" play.
Saves final model to pretrained_flows_nn.json for use in your JS front-end.
"""

import json, random, math, copy, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Tetris board size
BOARD_COLS, BOARD_ROWS = 10, 20

# Output JSON file for your front-end
FLOW_JSON_FILE = "pretrained_flows_nn.json"

TETROMINOES = {
    "I": [[0,0,0,0],
          [1,1,1,1],
          [0,0,0,0],
          [0,0,0,0]],
    "O": [[1,1],
          [1,1]],
    "T": [[0,1,0],
          [1,1,1],
          [0,0,0]],
    "S": [[0,1,1],
          [1,1,0],
          [0,0,0]],
    "Z": [[1,1,0],
          [0,1,1],
          [0,0,0]],
    "J": [[1,0,0],
          [1,1,1],
          [0,0,0]],
    "L": [[0,0,1],
          [1,1,1],
          [0,0,0]],
}
PIECES = list(TETROMINOES.keys())
PIECE_IDS = {k: i for i, k in enumerate(PIECES)}

def rotate(matrix):
    return [list(row) for row in zip(*matrix[::-1])]


class TetrisGame:
    """
    Tetris environment with partial line rewards plus survival bonus,
    and negative penalty for holes, top height, and bumpiness after each piece.
    """

    def __init__(self):
        self.cols = BOARD_COLS
        self.rows = BOARD_ROWS
        self.reset_game()

    def reset_game(self):
        self.board = [[0]*self.cols for _ in range(self.rows)]
        self.score = 0.0
        self.game_over = False
        # partially fill bottom row just to vary initial states
        row = [1]*self.cols
        empties = random.sample(range(self.cols), 3)
        for e in empties:
            row[e] = 0
        self.board[-1] = row
        self.spawn_piece()

    def spawn_piece(self):
        t_type = random.choice(PIECES)
        shape = copy.deepcopy(TETROMINOES[t_type])
        self.current_piece = {
            "type": t_type,
            "shape": shape,
            "x": self.cols//2,
            "y": 0
        }
        if self.collides(self.current_piece):
            self.game_over = True

    def collides(self, piece):
        for r, row in enumerate(piece["shape"]):
            for c, val in enumerate(row):
                if val:
                    x = piece["x"] + c
                    y = piece["y"] + r
                    if x<0 or x>= self.cols or y>= self.rows:
                        return True
                    if y>=0 and self.board[y][x]:
                        return True
        return False

    def clear_lines(self):
        newB = []
        for row in self.board:
            if not all(row):
                newB.append(row)
        cleared = self.rows - len(newB)
        while len(newB)< self.rows:
            newB.insert(0, [0]*self.cols)
        self.board = newB
        return cleared

    def get_top_height(self):
        for r in range(self.rows):
            if any(self.board[r]):
                return self.rows - r
        return 0

    def count_holes(self):
        holes = 0
        rows = len(self.board)
        cols = len(self.board[0])
        for c in range(cols):
            blockFound = False
            for r in range(rows):
                if self.board[r][c] == 1:
                    blockFound = True
                elif blockFound and self.board[r][c] == 0:
                    holes += 1
        return holes

    def count_bumpiness(self):
        rows = len(self.board)
        cols = len(self.board[0])
        heights = []
        for c in range(cols):
            h = 0
            for r in range(rows):
                if self.board[r][c]:
                    h = (rows - r)
                    break
            heights.append(h)
        bump = 0
        for c in range(cols-1):
            bump += abs(heights[c] - heights[c+1])
        return bump

    def lock_piece(self, piece):
        for r,row in enumerate(piece["shape"]):
            for c,val in enumerate(row):
                if val:
                    x = piece["x"]+ c
                    y = piece["y"]+ r
                    if 0<= x< self.cols and 0<= y< self.rows:
                        self.board[y][x] = 1
        return self.clear_lines()

    def get_moves(self):
        if self.game_over:
            return []
        cands = []
        t_type = self.current_piece["type"]
        rots = 1 if t_type=="O" else 4
        for rot in range(rots):
            shape = copy.deepcopy(TETROMINOES[t_type])
            for _ in range(rot):
                shape = rotate(shape)
            w = len(shape[0])
            for x in range(self.cols - w +1):
                testP = {
                    "type": t_type,
                    "shape": copy.deepcopy(shape),
                    "x": x,
                    "y": 0
                }
                if self.collides(testP):
                    continue
                while not self.collides(testP):
                    testP["y"]+=1
                testP["y"]-=1
                if testP["y"]>=0:
                    aK = f"r{rot}_x{x}"
                    depth = testP["y"]/ (self.rows-1)
                    cands.append((aK, testP, depth))
        return cands

    def apply_move(self, piece):
        lines = self.lock_piece(piece)
        # partial line reward:
        # e.g. 1 => +300, 2=>900,3=>1800,4=>3000
        # also survival bonus => +20
        # negative penalty => holes, topHeight, bumpiness
        lineReward = 0
        if lines==1: lineReward=300
        elif lines==2: lineReward=900
        elif lines==3: lineReward=1800
        elif lines==4: lineReward=3000
        # survival
        lineReward += 20

        # now penalty
        h = self.get_top_height()
        holes = self.count_holes()
        bump = self.count_bumpiness()
        # define some scale factors
        penalty = (holes*2.0 + h*0.5 + bump*1.0)
        # final delta
        delta = lineReward - penalty
        self.score += delta

        self.spawn_piece()
        if self.collides(self.current_piece):
            self.game_over = True

    def is_over(self):
        return self.game_over

    def get_final_reward(self):
        # no final penalty, just the accumulated partial
        return self.score


# -------------- Feature helpers & encode
def flatten_board(board):
    arr=[]
    for row in board:
        arr.extend(row)
    return arr

def count_holes(board):
    holes=0
    rows= len(board)
    cols= len(board[0])
    for c in range(cols):
        blockFound=False
        for r in range(rows):
            if board[r][c]==1:
                blockFound=True
            elif blockFound and board[r][c]==0:
                holes+=1
    return holes

def board_max_height(board):
    for r in range(len(board)):
        if any(board[r]):
            return (len(board)-r)
    return 0

def count_bumpiness(board):
    cols= len(board[0])
    rows= len(board)
    heights=[]
    for c in range(cols):
        h=0
        for r in range(rows):
            if board[r][c]:
                h= (rows-r)
                break
        heights.append(h)
    bump=0
    for c in range(cols-1):
        bump+= abs(heights[c]- heights[c+1])
    return bump

def get_heuristics(board):
    return [count_holes(board),
            board_max_height(board),
            count_bumpiness(board)]

def encode_state(board, piece):
    # board => 200
    flat= flatten_board(board)
    # pieceOneHot =>7
    oh= [0]*len(TETROMINOES)
    idx= PIECE_IDS[piece["type"]]
    oh[idx]=1
    # piece x,y => normalized
    px= piece["x"]/(BOARD_COLS-1)
    py= piece["y"]/(BOARD_ROWS-1)
    # heuristics => holes,bump, topH
    heur= get_heuristics(board)
    # total =>200 +7 +2 +3=212
    floats= flat + oh + [px,py] + heur
    return torch.tensor(floats, dtype=torch.float32)

def encode_action(aK):
    # "r2_x3"
    r,x= aK.split("_")
    rVal= int(r[1:])/3.0
    xVal= int(x[1:])/(BOARD_COLS-1)
    return torch.tensor([rVal,xVal], dtype=torch.float32)

STATE_DIM= 212
ACTION_DIM=2

class BigMLP(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(STATE_DIM+ACTION_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        )
    def forward(self,x):
        return self.net(x).squeeze(-1)

class NNReplayGFlowNet:
    """
    Trajectory-Balance-ish GFlowNet with replay, 
    partial line reward, negative penalty for holes, etc.
    """
    def __init__(self, lr=1e-4, replay_cap=50000, batch_size=64, eps_greedy=0.05):
        self.model= BigMLP()
        self.opt= optim.Adam(self.model.parameters(), lr=lr)
        self.logZ= 0.0
        self.replay= deque(maxlen=replay_cap)
        self.batch_size= batch_size
        self.eps= eps_greedy

    def sample_action(self, board, piece, cands):
        """Epsilon‑greedy + softmax sampling from a 1‑D flow vector."""
        if not cands:
            return None
        if random.random() < self.eps:
            return random.choice(cands)

        with torch.no_grad():
            # --- Encode state once ---
            b = torch.tensor(board, dtype=torch.float32) \
                    .view(1,1,BOARD_ROWS,BOARD_COLS)
            sb = self.encoder(b).squeeze(0)                 # (feat,)
            pi = self.piece_emb(torch.tensor([PIECE_IDS[piece["type"]]]))\
                     .squeeze(0)                              # (16,)
            s_emb = torch.cat([sb, pi], dim=-1)             # (feat+16,)

            # --- Build batches of size n_cands ---
            a_batch = torch.stack([encode_action(aK) for aK,_ in cands])  # (n,2)
            s_batch = s_emb.unsqueeze(0).repeat(len(cands), 1)            # (n,feat+16)

            # --- Forward through dueling net ---
            raw = self.flow_net(s_batch, a_batch)       # shape: (n,1)
            raw = raw.view(-1)                          # now (n,)
            raw = torch.clamp(raw, -20, 20)

            # --- Softmax with temperature ---
            exps = torch.exp((raw - raw.max()) / self.temperature)  # (n,)
            probs = exps / exps.sum()                               # (n,)

            # !!! DEBUG: print shapes !!!
            print(f"raw.shape={tuple(raw.shape)}, probs.shape={tuple(probs.shape)}")

            # --- Sample one index from this 1‑D distribution ---
            idx_tensor = torch.multinomial(probs, 1)  # shape: (1,)
            idx = idx_tensor.item()                  # scalar now safe

        return cands[idx]




    def store_trajectory(self, steps, final_r):
        self.replay.append( (steps, final_r) )

    def train_from_replay(self):
        if len(self.replay)< self.batch_size:
            return
        sampleBatch= random.sample(self.replay, self.batch_size)
        self.model.train()
        total_loss=0.0
        self.opt.zero_grad()
        for (steps,final_r) in sampleBatch:
            if final_r<0.01:
                final_r=0.01
            logR= math.log(final_r)
            bigInputs=[]
            offsets=[]
            start=0
            for (board,piece,chA,allA) in steps:
                sEnc= encode_state(board,piece)
                subT=[]
                cIdx=None
                for i, aK in enumerate(allA):
                    aEnc= encode_action(aK)
                    full= torch.cat([sEnc,aEnc])
                    subT.append(full)
                    if aK==chA:
                        cIdx=i
                length= len(subT)
                end= start+ length
                offsets.append( (start,end,cIdx) )
                bigInputs.extend(subT)
                start= end
            if not bigInputs:
                continue
            bigT= torch.stack(bigInputs)
            raw= self.model(bigT)
            raw= torch.clamp(raw, -20,20)
            sumLogp= []
            for (s,e,cI) in offsets:
                sub= raw[s:e]
                mx= sub.max()
                exps= torch.exp(sub- mx)
                sumE= exps.sum()
                lse= mx + torch.log(sumE)
                chosen= sub[cI]
                logp= chosen- lse
                sumLogp.append(logp)
            sumLogpTensor= torch.stack(sumLogp).sum()
            diff= sumLogpTensor + self.logZ - logR
            loss= diff**2
            loss.backward()
            total_loss+= loss.item()
        self.opt.step()
        # skip logZ update or do approximate
        return total_loss/ self.batch_size

    def export_json(self):
        weights={}
        for n,p in self.model.named_parameters():
            weights[n]= p.detach().cpu().tolist()
        data={
            "logZ": self.logZ,
            "weights": weights
        }
        with open(FLOW_JSON_FILE,"w") as f:
            json.dump(data,f,indent=2)

def train_tetris_nn(n_episodes=50000):
    agent= NNReplayGFlowNet(lr=1e-4,replay_cap=50000,batch_size=64,eps_greedy=0.05)
    rewards=[]
    best=0.0
    for ep in range(n_episodes):
        game= TetrisGame()
        game.reset_game()
        steps=[]
        while not game.is_over():
            cands= game.get_moves()
            if not cands:
                break
            chosen= agent.sample_action(game.board, game.current_piece, cands)
            if chosen is None:
                break
            (aK, pObj, _) = chosen
            allKeys= [cc[0] for cc in cands]
            bCopy= copy.deepcopy(game.board)
            pCopy= copy.deepcopy(game.current_piece)
            steps.append( (bCopy,pCopy,aK, allKeys) )
            game.apply_move(pObj)
        finalR= game.get_final_reward()
        if finalR> best:
            best= finalR
        rewards.append(finalR)
        agent.store_trajectory(steps, finalR)
        # do multiple replay steps
        for _ in range(5):
            agent.train_from_replay()

        if ep>0 and ep%100==0:
            avg= np.mean(rewards[-100:])
            best100= np.max(rewards[-100:])
            print(f"[{ep}] avg100={avg:.2f}, best100={best100:.2f}, bestGlobal={best:.2f}")

        # time slicing: save every 5000 episodes
        if (ep+1)%500==0:
            agent.export_json()

    agent.export_json()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward (with partial lines, hole penalty, etc.)")
    plt.title("Neural GFlowNet Tetris Training")
    plt.savefig("nn_gfn_tetris.png")

if __name__=="__main__":
    train_tetris_nn(50000)
