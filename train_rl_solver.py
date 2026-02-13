# train_rl_solver_corrected.py

"""
Training script for the RL Wordle Solver with PROPER train/test split.

BUGS FIXED:
1. Added train/test split (was evaluating on training data!)
2. Fixed word length mismatch issue
3. Added validation that words match expected length
4. Better error handling
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from game.rlSolver import RLSolver
from game.constants import DEFAULT_GAME_CONFIG
from game.util import read_to_lines



class WordleEnvironment:
    """
    Simple Wordle environment for training.
    """
    
    def __init__(self, word_list: List[str], n: int = 5):
        self.word_list = word_list
        self.n = n
        self.target_word = None
        
        # Validate all words are correct length
        invalid_words = [w for w in word_list if len(w) != n]
        if invalid_words:
            raise ValueError(f"Words must be {n} letters. Invalid: {invalid_words[:5]}")
        
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
        
        # BUGFIX: Validate guess length
        if len(guess) != self.n:
            raise ValueError(f"Guess '{guess}' is {len(guess)} letters, expected {self.n}")
        
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


def create_train_test_split(
    word_list: List[str],
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split word list into train and test sets.
    
    CRITICAL: Test set should be completely unseen during training!
    
    Args:
        word_list: Full list of words
        test_ratio: Fraction of words for test set (default 20%)
        random_seed: Random seed for reproducibility
    
    Returns:
        (train_words, test_words)
    """
    np.random.seed(random_seed)
    
    # Shuffle words
    shuffled_words = np.random.permutation(word_list).tolist()
    
    # Split
    test_size = int(len(shuffled_words) * test_ratio)
    test_words = shuffled_words[:test_size]
    train_words = shuffled_words[test_size:]
    
    print(f"Train/Test Split:")
    print(f"  Total words: {len(word_list)}")
    print(f"  Train words: {len(train_words)} ({100*(1-test_ratio):.0f}%)")
    print(f"  Test words:  {len(test_words)} ({100*test_ratio:.0f}%)")
    print(f"  Test ratio:  {test_ratio:.2f}")
    
    # Verify no overlap
    train_set = set(train_words)
    test_set = set(test_words)
    overlap = train_set & test_set
    
    if overlap:
        raise ValueError(f"Train/test overlap detected! {len(overlap)} words in both sets")
    
    return train_words, test_words


def train_rl_solver(
    train_config: Dict,
    test_config: Dict,
    num_episodes: int = 10000,
    eval_interval: int = 100,
    save_interval: int = 1000,
    model_save_path: str = "models/wordle_rl.pth"
):
    """
    Train the RL solver with proper train/test split.
    
    Args:
        train_config: Config with TRAINING words
        test_config: Config with TEST words (unseen during training!)
        num_episodes: Number of training episodes
        eval_interval: Evaluate every N episodes
        save_interval: Save model every N episodes
        model_save_path: Path to save the model
    """
    
    # Create directories
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialize solver with TRAINING config
    solver = RLSolver(config=train_config, verbose=False)
    
    # Initialize TRAINING environment
    train_env = WordleEnvironment(train_config['candidate_set'], n=solver.N)
    
    # Training metrics
    train_win_rates = []
    test_win_rates = []
    avg_guesses_train = []
    avg_guesses_test = []
    avg_rewards = []  # Track average reward
    epsilons = []  # Track epsilon values
    episodes_eval = []
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"State dim: {solver.state_dim}, Action dim: {solver.action_dim}")
    print(f"Device: {solver.device}")
    print(f"Training on {len(train_config['candidate_set'])} words")
    print(f"Testing on {len(test_config['candidate_set'])} words (unseen!)")
    print("-" * 60)
    
    for episode in range(1, num_episodes + 1):
        # Reset environment and solver
        train_env.reset()
        solver.candidate_set = list(train_config['candidate_set'])  # Use TRAIN words
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
            clue = train_env.get_clue(guess)
            
            # Check if won
            if train_env.check_win(guess):
                state = 1  # Won
                done = True
            elif solver.guess_number >= solver.MAX_GUESSES:
                state = 2  # Lost
                done = True
            
            # Incorporate feedback (will train automatically)
            solver.incorporate_guess_feedback(clue, state)
        
        # Periodic evaluation
        if episode % eval_interval == 0:
            # BUGFIX: Evaluate on BOTH train and test sets
            train_results = evaluate_solver(solver, train_config, num_games=100)
            test_results = evaluate_solver(solver, test_config, num_games=100)
            
            train_win_rates.append(train_results['win_rate'])
            test_win_rates.append(test_results['win_rate'])
            avg_guesses_train.append(train_results['avg_guesses'])
            avg_guesses_test.append(test_results['avg_guesses'])
            episodes_eval.append(episode)
            
            stats = solver.get_training_stats()
            
            # Track epsilon and average reward
            epsilons.append(stats['epsilon'])
            avg_rewards.append(stats['avg_reward_last_100'])
            
            print(f"Episode {episode}/{num_episodes}")
            print(f"  TRAIN Win Rate: {train_results['win_rate']:.2%}")
            print(f"  TEST Win Rate:  {test_results['win_rate']:.2%}")  # ← Key metric!
            print(f"  Train Avg Guesses: {train_results['avg_guesses']:.2f}")
            print(f"  Test Avg Guesses:  {test_results['avg_guesses']:.2f}")
            print(f"  Avg Reward (last 100): {stats['avg_reward_last_100']:.2f}")
            print(f"  Epsilon: {stats['epsilon']:.4f}")
            print(f"  Buffer Size: {stats['buffer_size']}")
            
            # BUGFIX: Check for overfitting
            if train_results['win_rate'] > 0.7 and test_results['win_rate'] < 0.4:
                print(f"  ⚠️  WARNING: Possible overfitting detected!")
            
            print("-" * 60)
        
        # Save model periodically
        if episode % save_interval == 0:
            solver.save_model(model_save_path)
            plot_training_progress(
                episodes_eval, 
                train_win_rates,
                test_win_rates,
                avg_guesses_train,
                avg_guesses_test,
                avg_rewards,
                epsilons,
                save_path=f"plots/training_progress_ep{episode}.png"
            )
    
    # Final save
    solver.save_model(model_save_path)
    print(f"\nTraining complete! Model saved to {model_save_path}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("\nTrain Set (1000 games):")
    train_final = evaluate_solver(solver, train_config, num_games=1000)
    print(f"  Win Rate: {train_final['win_rate']:.2%}")
    print(f"  Avg Guesses: {train_final['avg_guesses']:.2f}")
    print(f"  Distribution: {train_final['guess_distribution']}")
    
    print("\nTest Set (1000 games):")
    test_final = evaluate_solver(solver, test_config, num_games=1000)
    print(f"  Win Rate: {test_final['win_rate']:.2%}")
    print(f"  Avg Guesses: {test_final['avg_guesses']:.2f}")
    print(f"  Distribution: {test_final['guess_distribution']}")
    
    # Overfitting check
    gap = train_final['win_rate'] - test_final['win_rate']
    print(f"\nTrain-Test Gap: {gap:.2%}")
    if gap > 0.15:
        print("⚠️  WARNING: Model may be overfitting (>15% gap)")
    elif gap > 0.05:
        print("⚠️  Slight overfitting detected (>5% gap)")
    else:
        print("✓ Good generalization!")
    
    # Plot final results
    plot_training_progress(
        episodes_eval,
        train_win_rates,
        test_win_rates,
        avg_guesses_train,
        avg_guesses_test,
        avg_rewards,
        epsilons,
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
    
    BUGFIX: Now properly handles evaluation without training.
    
    Args:
        solver: Trained RLSolver
        config: Game configuration (can be train OR test config!)
        num_games: Number of games to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Save current epsilon and set to greedy
    original_epsilon = solver.epsilon
    solver.epsilon = 0.0  # Greedy evaluation - NO exploration
    
    # BUGFIX: Don't modify replay buffer, just disable training
    original_batch_size = solver.batch_size
    solver.batch_size = float('inf')  # Prevents training (buffer never "full enough")
    
    env = WordleEnvironment(config['candidate_set'], n=solver.N)
    
    wins = 0
    total_guesses = 0
    guess_distribution = {i: 0 for i in range(1, solver.MAX_GUESSES + 1)}
    guess_distribution['failed'] = 0
    
    for _ in range(num_games):
        # Reset - BUGFIX: Use the passed config, not solver's original config!
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
            
            # Incorporate feedback (won't train due to infinite batch_size)
            solver.incorporate_guess_feedback(clue, state)
    
    # Restore original settings
    solver.epsilon = original_epsilon
    solver.batch_size = original_batch_size
    
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
    train_win_rates: List[float],
    test_win_rates: List[float],
    train_avg_guesses: List[float],
    test_avg_guesses: List[float],
    avg_rewards: List[float],
    epsilons: List[float],
    save_path: str = "plots/training_progress.png"
):
    """
    Plot training progress with train/test comparison plus epsilon and reward tracking.
    
    Args:
        episodes: Episode numbers
        train_win_rates: Win rates on training set
        test_win_rates: Win rates on test set
        train_avg_guesses: Average guesses on training set
        test_avg_guesses: Average guesses on test set
        avg_rewards: Average reward per episode
        epsilons: Epsilon values over time
        save_path: Path to save the plot
    """
    if not episodes:  # No data to plot
        return
        
    # Create figure with 4 subplots (2x2 grid)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 24))
    
    # Subplot 1: Win rate comparison
    ax1.plot(episodes, train_win_rates, 'b-', linewidth=2, label='Train', marker='o', markersize=4)
    ax1.plot(episodes, test_win_rates, 'r-', linewidth=2, label='Test', marker='s', markersize=4)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title('Win Rate (Train vs Test)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
        
    # Subplot 2: Average guesses comparison
    ax2.plot(episodes, train_avg_guesses, 'b-', linewidth=2, label='Train', marker='o', markersize=4)
    ax2.plot(episodes, test_avg_guesses, 'r-', linewidth=2, label='Test', marker='s', markersize=4)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Guesses (when won)', fontsize=12)
    ax2.set_ylim(0.5, 6.5)
    ax2.set_title('Average Guesses (Train vs Test)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Average reward over time
    ax3.plot(episodes, avg_rewards, 'g-', linewidth=2.5, marker='D', markersize=5)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Average Reward (last 100 episodes)', fontsize=12)
    ax3.set_title('Average Reward', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        
    # Subplot 4: Epsilon decay over time
    ax4.plot(episodes, epsilons, 'purple', linewidth=2.5, marker='^', markersize=5)
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax4.set_title('Epsilon Decay', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, max(epsilons) * 1.1 if epsilons else 1])
    
    # Overall title
    fig.suptitle('Deep Q-Learning Training Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training progress plot saved to {save_path}")


if __name__ == "__main__":
    # Example usage with PROPER train/test split
    
    # Full word list (replace with actual Wordle words)
    guess_words = read_to_lines("data/de_leipzig_5_letter.txt")
    full_answer_words = read_to_lines("data/de_wiktionary_5_letter_shortlist.txt")
    
    # CRITICAL: Create train/test split
    train_words, test_words = create_train_test_split(
        full_answer_words,
        test_ratio=0.2,  # 20% for testing
        random_seed=42
    )
    
    # Separate configs for train and test
    train_config = {
        'candidate_set': train_words,
        'guess_set': guess_words,
        'max_guesses': 6
    }
    
    test_config = {
        'candidate_set': test_words,
        'guess_set': guess_words,
        'max_guesses': 6
    }
    
    print("\n" + "="*60)
    print("Starting RL Training with Train/Test Split")
    print("="*60 + "\n")
    
    # Train the solver
    trained_solver = train_rl_solver(
        train_config=train_config,
        test_config=test_config,
        num_episodes=10000,
        eval_interval=500,
        save_interval=1000,
        model_save_path="models/wordle_rl.pth"
    )
    
    print("\nTraining complete!")
