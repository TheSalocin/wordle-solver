# example_usage.py

"""
Simple example demonstrating how to use the RLSolver.
"""

from game.rlSolver import RLSolver
import numpy as np


def example_1_basic_usage():
    """Example 1: Basic initialization and single game."""
    
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Simple word list for demonstration
    word_list = [
        "crane", "slate", "saner", "arose", "irate",
        "stare", "snare", "crate", "trace", "brace",
        "react", "cater", "heart", "earth", "other"
    ]
    
    config = {
        'candidate_set': word_list,
        'guess_set': word_list,
        'max_guesses': 6
    }
    
    # Initialize solver
    solver = RLSolver(
        config=config,
        verbose=True,
        epsilon_start=0.5  # 50% exploration for demo
    )
    
    print(f"Solver initialized with {len(word_list)} words")
    print(f"State dimension: {solver.state_dim}")
    print(f"Action dimension: {solver.action_dim}")
    print()
    
    # Simulate a game (target word: "crane")
    target = "crane"
    print(f"Playing game with target word: '{target}'")
    print()
    
    done = False
    while not done:
        # Get guess
        guess = solver.choose_word()
        if guess is None:
            print("No more guesses available!")
            break
        
        # Generate clue (simplified)
        clue = generate_clue(guess, target)
        print(f"Clue: {clue}")
        
        # Check if won
        if guess == target:
            print(f"✓ Won in {solver.guess_number} guesses!")
            solver.incorporate_guess_feedback(clue, state=1)  # Won
            done = True
        elif solver.guess_number >= solver.MAX_GUESSES:
            print(f"✗ Lost! Target was '{target}'")
            solver.incorporate_guess_feedback(clue, state=2)  # Lost
            done = True
        else:
            solver.incorporate_guess_feedback(clue, state=0)  # Ongoing
        
        print()


def example_2_training_loop():
    """Example 2: Simple training loop."""
    
    print("=" * 60)
    print("Example 2: Training Loop")
    print("=" * 60)
    
    word_list = [
        "crane", "slate", "saner", "arose", "irate",
        "stare", "snare", "crate", "trace", "brace"
    ]
    
    config = {
        'candidate_set': word_list,
        'guess_set': word_list,
        'max_guesses': 6
    }
    
    solver = RLSolver(config=config, verbose=False)
    
    num_episodes = 100
    wins = 0
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Random target
        target = np.random.choice(word_list)
        
        # Reset solver for new game
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
                break
            
            clue = generate_clue(guess, target)
            
            if guess == target:
                wins += 1
                solver.incorporate_guess_feedback(clue, state=1)
                done = True
            elif solver.guess_number >= solver.MAX_GUESSES:
                solver.incorporate_guess_feedback(clue, state=2)
                done = True
            else:
                solver.incorporate_guess_feedback(clue, state=0)
        
        # Print progress
        if (episode + 1) % 20 == 0:
            win_rate = wins / (episode + 1)
            stats = solver.get_training_stats()
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Win Rate: {win_rate:.2%}, "
                  f"Epsilon: {stats['epsilon']:.3f}, "
                  f"Buffer: {stats['buffer_size']}")
    
    print(f"\nFinal Win Rate: {wins / num_episodes:.2%}")
    print(f"Final Epsilon: {solver.epsilon:.3f}")


def example_3_save_load():
    """Example 3: Save and load model."""
    
    print("=" * 60)
    print("Example 3: Save and Load Model")
    print("=" * 60)
    
    word_list = ["crane", "slate", "saner", "arose", "irate"]
    config = {
        'candidate_set': word_list,
        'guess_set': word_list,
        'max_guesses': 6
    }
    
    # Train a bit
    print("Training solver...")
    solver1 = RLSolver(config=config, verbose=False)
    
    # Quick training
    for _ in range(50):
        target = np.random.choice(word_list)
        solver1.candidate_set = list(config['candidate_set'])
        solver1.guess_number = 0
        solver1.guesses = []
        solver1.clues = []
        solver1.states = []
        solver1.current_state = None
        
        done = False
        while not done:
            guess = solver1.choose_word()
            if guess is None:
                break
            clue = generate_clue(guess, target)
            state = 1 if guess == target else (2 if solver1.guess_number >= 6 else 0)
            solver1.incorporate_guess_feedback(clue, state)
            if state != 0:
                done = True
    
    # Save model
    model_path = "example_model.pth"
    solver1.save_model(model_path)
    print(f"Model saved. Epsilon: {solver1.epsilon:.4f}")
    
    # Load into new solver
    print("\nLoading model into new solver...")
    solver2 = RLSolver(config=config, verbose=False)
    solver2.load_model(model_path)
    print(f"Model loaded. Epsilon: {solver2.epsilon:.4f}")
    
    # Verify they have same epsilon
    assert abs(solver1.epsilon - solver2.epsilon) < 1e-6
    print("✓ Model successfully saved and loaded!")
    
    # Clean up
    import os
    if os.path.exists(model_path):
        os.remove(model_path)


def example_4_state_encoding():
    """Example 4: Examine state encoding."""
    
    print("=" * 60)
    print("Example 4: State Encoding")
    print("=" * 60)
    
    word_list = ["crane", "slate", "arose"]
    config = {
        'candidate_set': word_list,
        'guess_set': word_list,
        'max_guesses': 6
    }
    
    solver = RLSolver(config=config, verbose=False)
    
    print(f"Initial state dimension: {solver.state_dim}")
    print(f"State components:")
    print(f"  - Letter frequencies: 26")
    print(f"  - Position info: {solver.N} × 26 = {solver.N * 26}")
    print(f"  - Excluded letters: 26")
    print(f"  - Confirmed letters: 26")
    print(f"  - Guesses remaining: 1")
    print(f"  - Clue history: {solver.MAX_GUESSES} × {solver.N} × 3 = {solver.MAX_GUESSES * solver.N * 3}")
    print()
    
    # Encode initial state
    state = solver._encode_state()
    print(f"Encoded state shape: {state.shape}")
    print(f"First 10 values: {state[:10]}")
    print()
    
    # Make a guess and see how state changes
    guess = solver.choose_word()
    print(f"Made guess: '{guess}'")
    
    # Simulate feedback
    target = "arose"
    clue = generate_clue(guess, target)
    print(f"Clue: {clue}")
    
    solver.incorporate_guess_feedback(clue, state=0)
    
    # Encode new state
    new_state = solver._encode_state()
    print(f"New state shape: {new_state.shape}")
    
    # Show what changed
    diff = np.abs(state - new_state).sum()
    print(f"State difference (L1 norm): {diff:.2f}")
    print(f"Remaining candidates: {len(solver.candidate_set)}")


def generate_clue(guess: str, target: str) -> list:
    """
    Generate clue for a guess.
    
    Returns:
        List of integers: 0=gray, 1=yellow, 2=green
    """
    n = len(target)
    clue = [0] * n
    target_chars = list(target)
    guess_chars = list(guess)
    
    # First pass: mark greens
    for i in range(n):
        if guess_chars[i] == target_chars[i]:
            clue[i] = 2
            target_chars[i] = None
            guess_chars[i] = None
    
    # Second pass: mark yellows
    for i in range(n):
        if guess_chars[i] is not None and guess_chars[i] in target_chars:
            clue[i] = 1
            target_chars[target_chars.index(guess_chars[i])] = None
    
    return clue


if __name__ == "__main__":
    # Run all examples
    example_1_basic_usage()
    print("\n" * 2)
    
    example_2_training_loop()
    print("\n" * 2)
    
    example_3_save_load()
    print("\n" * 2)
    
    example_4_state_encoding()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
