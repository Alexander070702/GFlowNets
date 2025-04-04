#!/usr/bin/env python3
import argparse
import time
import json
import random
import copy
import os
import math

# Import the TetrisGame and TrajectoryBalanceAgent classes from our app module.
from app import TetrisGame, TrajectoryBalanceAgent

def safe_save_json(data, filepath):
    """
    Write `data` to `filepath` as JSON atomically.
    This function writes to a temporary file first and then renames it,
    ensuring that the file is always in a valid state (i.e., not partially written).
    """
    temp_path = filepath + ".tmp"  # Append .tmp to create a temporary filename.
    with open(temp_path, "w") as f:
        json.dump(data, f)         # Write data to the temporary file.
    os.replace(temp_path, filepath)  # Atomically replace the target file with the temporary file.

def simulate_episode(game, agent):
    """
    Run one complete Tetris game (episode) until game over.
    During the game, record each (state, action) pair in the trajectory.
    After the game ends, compute the final reward and use it to update the agent.
    
    Returns:
        final_reward (float): The final reward obtained at the end of the episode.
    """
    trajectory = []  # This list will store the sequence of (state_key, action_key) tuples.
    
    # Loop until the game signals that it is over.
    while not game.is_over():
        state_key = game.get_state_key()       # Get a unique key representing the current state.
        cands = game.get_terminal_moves()        # Compute candidate terminal moves for the current piece.
        
        if not cands:
            # If there are no candidate moves, mark the game as over.
            game.game_over = True
            break
        
        # Sample an action from the agent using the Trajectory Balance policy.
        # This method returns a candidate and its probability (we don't use the probability here).
        selected_action, _p = agent.sample_action(state_key, cands)
        
        # Record the chosen (state, action) pair in the trajectory for later updates.
        trajectory.append((state_key, selected_action['action_key']))
        
        # "Place" the piece in the game: copy the selected candidate's piece configuration
        # to the current piece and then lock it in place.
        game.current_piece = copy.deepcopy(selected_action['piece'])
        game.lock_piece()  # This method locks the piece in place, clears lines if necessary, and spawns a new piece.
        
        if game.is_over():
            # If locking the piece ended the game, exit the loop.
            break
        
        # The loop continues until no moves remain (i.e. the game is over).
    
    # After the game has ended, compute the final reward.
    final_reward = game.get_final_reward()
    # Update the agent's parameters using the trajectory and the final reward.
    # The update method adjusts the log flows and normalization constant so that the
    # probability of the trajectory aligns with the observed reward.
    agent.update_trajectory(trajectory, final_reward)
    
    # Return the final reward for reporting purposes.
    return final_reward

def pretrain(num_episodes, checkpoint_interval, checkpoint_file, lr):
    """
    Run the pretraining loop for a given number of episodes.
    
    Parameters:
        num_episodes (int): Total number of episodes (games) to simulate.
        checkpoint_interval (int): How often (in episodes) to save the agent's parameters.
        checkpoint_file (str): File path where the agent's parameters will be saved.
        lr (float): Learning rate for the Trajectory Balance updates.
    """
    # Create a new agent with the specified learning rate.
    agent = TrajectoryBalanceAgent(lr=lr)
    
    # Initialize a variable to accumulate total reward over all episodes.
    total_reward = 0.0
    # Record the starting time to compute elapsed time during training.
    start_time = time.time()

    # Loop over the desired number of episodes.
    for episode in range(1, num_episodes+1):
        # Create a new game for each episode.
        # We use a smaller board (6 columns by 10 rows) to make training faster and simpler.
        game = TetrisGame(cols=6, rows=10)
        # Simulate one complete game episode.
        reward = simulate_episode(game, agent)
        # Add the episode's reward to our running total.
        total_reward += reward

        # Every 'checkpoint_interval' episodes, save the agent's parameters to a file.
        if episode % checkpoint_interval == 0:
            avg_reward = total_reward / episode
            elapsed = time.time() - start_time
            # Print progress information.
            print(f"Episode {episode}/{num_episodes} | Last Reward: {reward} "
                  f"| Avg Reward: {avg_reward:.2f} | Elapsed: {elapsed:.2f}s")
            # Prepare data to save: both the log flows and the normalization constant.
            data = {
                "log_flows": agent.log_flows,
                "logZ": agent.logZ
            }
            try:
                safe_save_json(data, checkpoint_file)
                print(f"Checkpoint saved to '{checkpoint_file}'.")
            except Exception as e:
                print("Error saving checkpoint:", e)

    # After all episodes, print the final average reward.
    print("Training complete. Average reward:", total_reward / num_episodes)
    
    # Save the final agent parameters.
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
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(description="Pretrain the GFlowNet (Trajectory Balance) for Tetris.")
    parser.add_argument("--episodes", type=int, default=200000,
                        help="Number of episodes (games) to train.")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="How often to save checkpoint.")
    parser.add_argument("--checkpoint_file", type=str, default="pretrained_flows_tb.json",
                        help="File to save pretrained flows.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for TB updates.")
    args = parser.parse_args()

    # Start the pretraining process using the provided arguments.
    pretrain(args.episodes, args.checkpoint_interval, args.checkpoint_file, args.lr)
