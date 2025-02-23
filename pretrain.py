#!/usr/bin/env python3
import argparse
import time
import json
import random
import copy
import os
import math

from app import TetrisGame, TrajectoryBalanceAgent

def safe_save_json(data, filepath):
    """
    Write `data` to `filepath` as JSON atomically.
    1) Write to filepath.tmp
    2) Close and rename to filepath
    """
    temp_path = filepath + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f)
    os.replace(temp_path, filepath)

def simulate_episode(game, agent):
    """
    Run one Tetris game until game over, with a single final reward at the end.
    We'll store the entire sequence of (state, action) in 'trajectory'.
    Then do TB update once the game is over.
    """
    trajectory = []
    while not game.is_over():
        state_key = game.get_state_key()
        cands = game.get_terminal_moves()
        if not cands:
            # No moves => game over
            game.game_over = True
            break

        # Sample action from the TB policy
        selected_action, _p = agent.sample_action(state_key, cands)

        # Record
        trajectory.append((state_key, selected_action['action_key']))

        # "place" that final piece
        game.current_piece = copy.deepcopy(selected_action['piece'])
        game.lock_piece()  # lock and spawn next piece
        if game.is_over():
            break

        # The environment loops until game over.

    # Now the game ended => final reward
    final_reward = game.get_final_reward()
    agent.update_trajectory(trajectory, final_reward)

    return final_reward

def pretrain(num_episodes, checkpoint_interval, checkpoint_file, lr):
    agent = TrajectoryBalanceAgent(lr=lr)
    total_reward = 0.0
    start_time = time.time()

    for episode in range(1, num_episodes+1):
        game = TetrisGame(cols=6, rows=10)
        reward = simulate_episode(game, agent)
        total_reward += reward

        if episode % checkpoint_interval == 0:
            avg_reward = total_reward / episode
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{num_episodes} | Last Reward: {reward} "
                  f"| Avg Reward: {avg_reward:.2f} | Elapsed: {elapsed:.2f}s")
            # Save the agent's parameters
            data = {
                "log_flows": agent.log_flows,
                "logZ": agent.logZ
            }
            try:
                safe_save_json(data, checkpoint_file)
                print(f"Checkpoint saved to '{checkpoint_file}'.")
            except Exception as e:
                print("Error saving checkpoint:", e)

    print("Training complete. Average reward:", total_reward / num_episodes)
    # Final save
    data = {
        "log_flows": agent.log_flows,
        "logZ": agent.logZ
    }
    try:
        safe_save_json(data, checkpoint_file)
        print(f"Pretrained flows saved to '{checkpoint_file}'.")
    except Exception as e:
        print("Error saving final pretrained flows:", e)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Pretrain the GFlowNet (Trajectory Balance) for Tetris.")
    parser.add_argument("--episodes", type=int, default=200000,
                        help="Number of episodes (games) to train.")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="How often to save checkpoint.")
    parser.add_argument("--checkpoint_file", type=str, default="pretrained_flows_tb.json",
                        help="File to save pretrained flows.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for TB updates.")
    args = parser.parse_args()

    pretrain(args.episodes, args.checkpoint_interval, args.checkpoint_file, args.lr)
