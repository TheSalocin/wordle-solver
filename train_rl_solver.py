# train_rl_solver.py

"""
Training script for the RL Wordle Solver.

This script demonstrates how to:
1. Initialize the RLSolver
2. Run training episodes
3. Save/load models
4. Evaluate performance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from game.rlSolver import RLSolver
from game.constants import DEFAULT_GAME_CONFIG


class WordleEnvironment:
    """
    Simple Wordle environment for training.
    """
    
    def __init__(self, word_list: List[str], n: int = 5):
        self.word_list = word_list
        self.n = n
        self.target_word = None
        self.reset()
    
    def reset(self) -> None:
        """Reset environment with new target word."""
        self.target_word = np.random.choice(self.word_list)
    
    def get_clue(self, guess: str) -> List[int]:
        """
        Generate clue for a guess.
        
        Returns:
            List of integers: 0=gray, 1=yellow, 2=green
        """
        if not self.target_word:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        clue = [0] * self.n
        target_chars = list(self.target_word)
        guess_chars = list(guess)
        
        # First pass: mark greens
        for i in range(self.n):
            if guess_chars[i] == target_chars[i]:
                clue[i] = 2
                target_chars[i] = None  # Mark as used
                guess_chars[i] = None
        
        # Second pass: mark yellows
        for i in range(self.n):
            if guess_chars[i] is not None and guess_chars[i] in target_chars:
                clue[i] = 1
                # Remove first occurrence
                target_chars[target_chars.index(guess_chars[i])] = None
        
        return clue
    
    def check_win(self, guess: str) -> bool:
        """Check if guess matches target."""
        return guess == self.target_word


def train_rl_solver(
    config: Dict,
    num_episodes: int = 10000,
    eval_interval: int = 100,
    save_interval: int = 1000,
    model_save_path: str = "models/wordle_rl.pth"
):
    """
    Train the RL solver.
    
    Args:
        config: Game configuration
        num_episodes: Number of training episodes
        eval_interval: Evaluate every N episodes
        save_interval: Save model every N episodes
        model_save_path: Path to save the model
    """
    
    # Create directories
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialize solver
    solver = RLSolver(config=config, verbose=False)
    
    # Initialize environment
    env = WordleEnvironment(config['candidate_set'], n=solver.N)
    
    # Training metrics
    win_rates = []
    avg_guesses = []
    episodes_eval = []
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"State dim: {solver.state_dim}, Action dim: {solver.action_dim}")
    print(f"Device: {solver.device}")
    print("-" * 60)
    
    for episode in range(1, num_episodes + 1):
        # Reset environment and solver
        env.reset()
        solver.candidate_set = list(config['candidate_set'])
        solver.guess_number = 0
        solver.guesses = []
        solver.clues = []
        solver.states = []
        solver.current_state = None
        solver.episode_reward = 0
        
        # Play episode
        done = False
        state = 0  # 0: ongoing, 1: won, 2: lost
        
        while not done:
            # Get guess from solver
            guess = solver.choose_word()
            
            if guess is None:
                # Max guesses reached
                state = 2  # Lost
                done = True
                solver.incorporate_guess_feedback([0] * solver.N, state)
                break
            
            # Get feedback from environment
            clue = env.get_clue(guess)
            
            # Check if won
            if env.check_win(guess):
                state = 1  # Won
                done = True
            elif solver.guess_number >= solver.MAX_GUESSES:
                state = 2  # Lost
                done = True
            
            # Incorporate feedback (will train automatically)
            solver.incorporate_guess_feedback(clue, state)
        
        # Periodic evaluation
        if episode % eval_interval == 0:
            eval_results = evaluate_solver(solver, config, num_games=100)
            win_rates.append(eval_results['win_rate'])
            avg_guesses.append(eval_results['avg_guesses'])
            episodes_eval.append(episode)
            
            stats = solver.get_training_stats()
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Win Rate: {eval_results['win_rate']:.2%}")
            print(f"  Avg Guesses: {eval_results['avg_guesses']:.2f}")
            print(f"  Avg Reward (last 100): {stats['avg_reward_last_100']:.2f}")
            print(f"  Epsilon: {stats['epsilon']:.4f}")
            print(f"  Buffer Size: {stats['buffer_size']}")
            print("-" * 60)
        
        # Save model periodically
        if episode % save_interval == 0:
            solver.save_model(model_save_path)
            plot_training_progress(
                episodes_eval, 
                win_rates, 
                avg_guesses, 
                save_path=f"plots/training_progress_ep{episode}.png"
            )
    
    # Final save
    solver.save_model(model_save_path)
    print(f"\nTraining complete! Model saved to {model_save_path}")
    
    # Final evaluation
    print("\nFinal Evaluation (1000 games):")
    final_results = evaluate_solver(solver, config, num_games=1000)
    print(f"  Win Rate: {final_results['win_rate']:.2%}")
    print(f"  Avg Guesses: {final_results['avg_guesses']:.2f}")
    print(f"  Guess Distribution: {final_results['guess_distribution']}")
    
    # Plot final results
    plot_training_progress(
        episodes_eval, 
        win_rates, 
        avg_guesses, 
        save_path="plots/final_training_progress.png"
    )
    
    return solver


def evaluate_solver(
    solver: RLSolver,
    config: Dict,
    num_games: int = 100
) -> Dict:
    """
    Evaluate solver performance.
    
    Args:
        solver: Trained RLSolver
        config: Game configuration
        num_games: Number of games to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Save current epsilon and set to greedy
    original_epsilon = solver.epsilon
    solver.epsilon = 0.0  # Greedy evaluation
    
    # Save current replay buffer and create a temporary one
    # This prevents training during evaluation
    original_buffer = solver.replay_buffer
    from game.rlSolver import ReplayBuffer
    solver.replay_buffer = ReplayBuffer(capacity=1000)  # Small temporary buffer
    
    env = WordleEnvironment(config['candidate_set'], n=solver.N)
    
    wins = 0
    total_guesses = 0
    guess_distribution = {i: 0 for i in range(1, solver.MAX_GUESSES + 1)}
    guess_distribution['failed'] = 0
    
    for _ in range(num_games):
        # Reset
        env.reset()
        solver.candidate_set = list(config['candidate_set'])
        solver.guess_number = 0
        solver.guesses = []
        solver.clues = []
        solver.states = []
        solver.current_state = None
        
        # Play game
        done = False
        while not done:
            guess = solver.choose_word()
            
            if guess is None:
                guess_distribution['failed'] += 1
                break
            
            clue = env.get_clue(guess)
            
            if env.check_win(guess):
                wins += 1
                total_guesses += solver.guess_number
                guess_distribution[solver.guess_number] += 1
                done = True
                state = 1
            elif solver.guess_number >= solver.MAX_GUESSES:
                guess_distribution['failed'] += 1
                done = True
                state = 2
            else:
                state = 0
            
            # Incorporate feedback (won't train much due to greedy epsilon)
            solver.incorporate_guess_feedback(clue, state)
    
    # Restore original settings
    solver.epsilon = original_epsilon
    solver.replay_buffer = original_buffer
    
    win_rate = wins / num_games
    avg_guesses = total_guesses / wins if wins > 0 else 0
    
    return {
        'win_rate': win_rate,
        'avg_guesses': avg_guesses,
        'guess_distribution': guess_distribution,
        'num_games': num_games
    }


def plot_training_progress(
    episodes: List[int],
    win_rates: List[float],
    avg_guesses: List[float],
    save_path: str = "plots/training_progress.png"
):
    """
    Plot training progress.
    
    Args:
        episodes: Episode numbers
        win_rates: Win rates at each evaluation
        avg_guesses: Average guesses at each evaluation
        save_path: Path to save the plot
    """
    if not episodes:  # No data to plot
        return
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Win rate
    ax1.plot(episodes, win_rates, 'b-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Training Progress: Win Rate')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Average guesses
    ax2.plot(episodes, avg_guesses, 'r-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Guesses (when won)')
    ax2.set_title('Training Progress: Average Guesses')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training progress plot saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    
    # Simple word list for testing (replace with actual word list)
    word_list = [
        "crane", "slate", "saner", "arose", "irate",
        "stare", "snare", "crate", "trace", "brace",
        "react", "cater", "heart", "earth", "other",
        "their", "while", "about", "place", "there"
    ]
    
    config = {
        'candidate_set': word_list,
        'guess_set': word_list,
        'max_guesses': 6
    }
    
    # Train the solver
    trained_solver = train_rl_solver(
        config=config,
        num_episodes=1000,  # Reduced for quick testing
        eval_interval=100,
        save_interval=500,
        model_save_path="models/wordle_rl.pth"
    )
    
    print("\nTraining complete!")
