# train_tetris_gflownet.py
# Python script to train a Tetris-playing agent using GFlowNet (Trajectory Balance) methodology

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Optional: if using gym-tetris
# pip install gym-tetris nes-py
try:
    import gym_tetris
    from nes_py.wrappers import JoypadSpace
    from gym_tetris.actions import SIMPLE_MOVEMENT
except ImportError:
    raise ImportError("Please install 'gym-tetris' and 'nes-py' to use this script: pip install gym-tetris nes-py")

# ====================== GFlowNet Model ======================
class GFlowNetModel(nn.Module):
    def __init__(self, state_dim, action_space, hidden_sizes=[512, 256]):
        super(GFlowNetModel, self).__init__()
        self.state_dim = state_dim
        self.action_space = action_space
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.flow_head = nn.Linear(hidden_sizes[1], action_space)
        # logZ as learnable parameter
        self.logZ = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        log_flows = self.flow_head(x)  # outputs unnormalized log flows for each action
        return log_flows, self.logZ

# ====================== Utilities ======================
def sample_trajectory(env, model, device):
    states, actions = [], []
    state = env.reset()
    done = False
    while not done:
        states.append(state)
        state_tensor = torch.from_numpy(state).float().to(device)
        log_flows, _ = model(state_tensor.unsqueeze(0))
        flows = torch.exp(log_flows).detach().cpu().numpy().flatten()
        probs = flows / flows.sum()
        action_idx = np.random.choice(len(probs), p=probs)
        actions.append(action_idx)
        next_state, reward, done, info = env.step(action_idx)
        state = next_state
    return states, actions, info['score'] if 'score' in info else reward

# ====================== Training Loop ======================
def train_gflownet(
    env_name='TetrisA-v0', episodes=5000, batch_size=16,
    lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):

    # Initialize environment
    env = gym_tetris.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    obs_space = env.observation_space.shape
    # Flattened observation dimension (e.g. 240x256x3 for RGB) -- adjust for grayscale or preprocess
    state_dim = int(np.prod(obs_space))
    action_space = env.action_space.n

    # Model and optimizer
    model = GFlowNetModel(state_dim, action_space).to(device)
    optimizer = optim.Adam(list(model.parameters()), lr=lr)

    for ep in range(1, episodes+1):
        # Sample a batch of trajectories
        batch_loss = 0.0
        for _ in range(batch_size):
            states, actions, reward = sample_trajectory(env, model, device)
            logR = np.log(max(reward, 1e-6))
            sum_logf = 0.0
            for s, a in zip(states, actions):
                s_t = torch.from_numpy(s).float().to(device).unsqueeze(0)
                log_flows, logZ = model(s_t)
                sum_logf += log_flows[0, a]
            loss = (sum_logf - logR - logZ).pow(2)
            batch_loss += loss
        # Backprop and update parameters
        optimizer.zero_grad()
        (batch_loss / batch_size).backward()
        optimizer.step()

        # Logging
        if ep % 100 == 0:
            print(f"Episode {ep}/{episodes} | Loss: {(batch_loss / batch_size).item():.4f}")
            # Optionally evaluate
    # Save the trained model
    torch.save(model.state_dict(), 'gflownet_tetris.pth')
    print("Training complete, model saved as 'gflownet_tetris.pth'")

if __name__ == '__main__':
    train_gflownet()
