import json
import random
from tqdm import trange
import torch
from gflownet.envs.tetris import Tetris

# ────────────────────────────────────────────────────────────────────────────────
# 1) Override get_state_key to match exactly what the JS does via state2readable()
# ────────────────────────────────────────────────────────────────────────────────
setattr(Tetris, "get_state_key", lambda self: self.state2readable())


# ────────────────────────────────────────────────────────────────────────────────
# 2) Trajectory Balance Agent
# ────────────────────────────────────────────────────────────────────────────────
class TrajectoryBalanceAgent:
    def __init__(self, lr=0.01):
        self.log_flows = {}    # state_key -> { action_key: log_flow }
        self.logZ = 0.0
        self.lr = lr

    def _ensure_action_exists(self, state_key, action_key):
        self.log_flows.setdefault(state_key, {})
        if action_key not in self.log_flows[state_key]:
            init_val = 0.5 + random.random()
            self.log_flows[state_key][action_key] = torch.log(torch.tensor(init_val))

    def sample_action(self, state_key, candidates):
        # ensure entries
        for c in candidates:
            self._ensure_action_exists(state_key, c['action_key'])
        logs = torch.tensor([
            self.log_flows[state_key][c['action_key']]
            for c in candidates
        ])
        probs = torch.softmax(logs, dim=0).tolist()
        idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
        return candidates[idx], probs[idx]

    def get_log_p_action(self, state_key, action_key):
        logs = torch.stack(list(self.log_flows[state_key].values()))
        denom = torch.logsumexp(logs, dim=0)
        return self.log_flows[state_key][action_key] - denom

    def update_trajectory(self, trajectory, final_reward):
        if final_reward <= 0:
            final_reward = 1e-2
        logR = torch.log(torch.tensor(final_reward))
        sum_logp = torch.tensor(0.0)
        for s, a in trajectory:
            sum_logp += self.get_log_p_action(s, a)
        target = logR - self.logZ
        diff = sum_logp - target

        # update logZ
        self.logZ += self.lr * diff.item()

        # update flows
        for s, a in trajectory:
            self.log_flows[s][a] -= self.lr * diff


# ────────────────────────────────────────────────────────────────────────────────
# 3) Training Loop
# ────────────────────────────────────────────────────────────────────────────────
def train(agent, env, episodes=20000):
    for _ in trange(episodes):
        env.reset()
        trajectory = []
        state_key = env.get_state_key()

        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions or valid_actions == [env.eos]:
                break

            candidates = []
            for act in valid_actions:
                if act == env.eos:
                    continue
                _, rotation, col = act
                action_key = f"r{rotation}_x{col}"
                candidates.append({
                    "action_key":  action_key,
                    "action_tuple": act
                })

            if not candidates:
                break

            cand, _ = agent.sample_action(state_key, candidates)
            trajectory.append((state_key, cand["action_key"]))

            _, _, valid = env.step(cand["action_tuple"])
            if not valid or env.done:
                break

            state_key = env.get_state_key()

        final_reward = env.n_actions
        agent.update_trajectory(trajectory, final_reward)

    return agent


# ────────────────────────────────────────────────────────────────────────────────
# 4) Main: train & dump JSON
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env   = Tetris(device="cpu")
    agent = TrajectoryBalanceAgent(lr=0.01)
    agent = train(agent, env, episodes=5000)

    out = {"logZ": agent.logZ, "log_flows": {}}
    for sk, d in agent.log_flows.items():
        out["log_flows"][sk] = { ak: v.item() for ak, v in d.items() }

    with open("pretrained_flows_tb.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("Saved pretrained_flows_tb.json")
