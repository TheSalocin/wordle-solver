# game/rlSolver.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Set, Optional
from game.baseSolver import Solver
from game.util import get_n_from_word_set
from game.constants import DEFAULT_GAME_CONFIG


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class CharacterMapper:
    """
    Maps characters to indices, supporting both regular letters and German umlauts.
    
    Mapping:
    a-z: indices 0-25
    ä: index 26
    ö: index 27
    ü: index 28
    """
    
    def __init__(self):
        # Regular letters a-z
        self.char_to_idx = {chr(ord('a') + i): i for i in range(26)}
        
        # German umlauts
        self.char_to_idx['ä'] = 26
        self.char_to_idx['ö'] = 27
        self.char_to_idx['ü'] = 28
        
        # Reverse mapping
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.num_chars = 29
    
    def char_to_index(self, char: str) -> int:
        """
        Convert character to index.
        
        Args:
            char: Single character (a-z, ä, ö, ü)
            
        Returns:
            Index (0-28)
            
        Raises:
            ValueError: If character is not supported
        """
        char_lower = char.lower()
        if char_lower not in self.char_to_idx:
            raise ValueError(f"Unsupported character: '{char}'. Supported: a-z, ä, ö, ü")
        return self.char_to_idx[char_lower]
    
    def index_to_char(self, idx: int) -> str:
        """
        Convert index to character.
        
        Args:
            idx: Index (0-28)
            
        Returns:
            Character
        """
        if idx not in self.idx_to_char:
            raise ValueError(f"Invalid index: {idx}. Valid range: 0-28")
        return self.idx_to_char[idx]
    
    def get_num_characters(self) -> int:
        """Get total number of supported characters."""
        return self.num_chars


class WordleQNetwork(nn.Module):
    """
    Deep Q-Network for Wordle.
    
    State representation includes:
    - One-hot encoding of previous guesses and their feedback
    - Available letter information
    - Position constraints
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super(WordleQNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for stable training."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class RLSolver(Solver):
    """
    Reinforcement Learning Solver for Wordle using Deep Q-Learning.
    Supports German umlauts (ä, ö, ü).
    
    This extends the base Solver class with:
    - State encoding from game history
    - Q-network for action selection
    - Experience replay for training
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        config: Dict[str, str] = DEFAULT_GAME_CONFIG,
        manual: bool = False,
        verbose: bool = True,
        # RL-specific parameters
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.999,
        batch_size: int = 64,
        buffer_size: int = 100000,
        target_update_freq: int = 1000,
        device: str = None
    ):
        super().__init__(config, manual, verbose)
        
        # Initialize character mapper for umlauts
        self.char_mapper = CharacterMapper()
        self.num_chars = self.char_mapper.get_num_characters()  # 29 (a-z + ä, ö, ü)
        
        # RL hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create word-to-index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(self.guess_set)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # State and action dimensions
        self.state_dim = self._calculate_state_dim()
        self.action_dim = len(self.guess_set)
        
        # Initialize Q-networks (main and target)
        self.q_network = WordleQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = WordleQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.training_steps = 0
        self.episode_rewards = []
        self.losses = []
        
        # Current episode tracking
        self.current_state = None
        self.episode_reward = 0
        
    def _calculate_state_dim(self) -> int:
        """
        Calculate state dimension based on:
        - Letter frequencies (29 characters: a-z + ä, ö, ü)
        - Position information (N positions × 29 characters)
        - Excluded letters (29 characters)
        - Confirmed letters (29 characters)
        - Number of guesses remaining (1 scalar)
        - Previous clue encoding (MAX_GUESSES × N × 3 states)
        """
        letter_freq_dim = self.num_chars
        position_info_dim = self.N * self.num_chars
        excluded_dim = self.num_chars
        confirmed_dim = self.num_chars
        guesses_remaining_dim = 1
        clue_history_dim = self.MAX_GUESSES * self.N * 3
        
        total_dim = (letter_freq_dim + position_info_dim + excluded_dim + 
                    confirmed_dim + guesses_remaining_dim + clue_history_dim)
        
        return total_dim
    
    def _encode_state(self) -> np.ndarray:
        """
        Encode the current game state into a feature vector.
        
        Returns:
            numpy array representing the current state
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        idx = 0
        
        # 1. Letter frequency from candidate set (29 features: a-z + ä, ö, ü)
        letter_counts = np.zeros(self.num_chars)
        for word in self.candidate_set:
            for char in word:
                try:
                    char_idx = self.char_mapper.char_to_index(char)
                    letter_counts[char_idx] += 1
                except ValueError as e:
                    if self.verbose:
                        print(f"Warning: {e} in word '{word}'")
                    continue
        if len(self.candidate_set) > 0:
            letter_counts /= len(self.candidate_set)
        state[idx:idx+self.num_chars] = letter_counts
        idx += self.num_chars
        
        # 2. Position-specific letter information (N × 29 features)
        position_info = np.zeros((self.N, self.num_chars))
        for word in self.candidate_set:
            for pos, char in enumerate(word):
                try:
                    char_idx = self.char_mapper.char_to_index(char)
                    position_info[pos][char_idx] += 1
                except ValueError:
                    continue
        if len(self.candidate_set) > 0:
            position_info /= len(self.candidate_set)
        state[idx:idx+(self.N*self.num_chars)] = position_info.flatten()
        idx += (self.N * self.num_chars)
        
        # 3. Excluded letters (29 features)
        excluded = np.zeros(self.num_chars)
        for clue_idx, clue in enumerate(self.clues):
            guess = self.guesses[clue_idx]
            for i, c in enumerate(clue):
                if c == 0:  # Gray letter
                    try:
                        char_idx = self.char_mapper.char_to_index(guess[i])
                        excluded[char_idx] = 1
                    except ValueError:
                        continue
        state[idx:idx+self.num_chars] = excluded
        idx += self.num_chars
        
        # 4. Confirmed letters (29 features)
        confirmed = np.zeros(self.num_chars)
        for clue_idx, clue in enumerate(self.clues):
            guess = self.guesses[clue_idx]
            for i, c in enumerate(clue):
                if c == 2:  # Green letter
                    try:
                        char_idx = self.char_mapper.char_to_index(guess[i])
                        confirmed[char_idx] = 1
                    except ValueError:
                        continue
        state[idx:idx+self.num_chars] = confirmed
        idx += self.num_chars
        
        # 5. Guesses remaining (1 feature)
        state[idx] = (self.MAX_GUESSES - self.guess_number) / self.MAX_GUESSES
        idx += 1
        
        # 6. Previous clue history (MAX_GUESSES × N × 3 features)
        clue_history = np.zeros((self.MAX_GUESSES, self.N, 3))
        for guess_idx, clue in enumerate(self.clues):
            for pos, clue_val in enumerate(clue):
                if clue_val < 3:  # Valid clue values: 0, 1, 2
                    clue_history[guess_idx][pos][clue_val] = 1
        state[idx:idx+(self.MAX_GUESSES*self.N*3)] = clue_history.flatten()
        
        return state
    
    def _get_valid_actions(self) -> List[int]:
        """
        Get list of valid action indices based on current candidate set.
        
        Returns:
            List of valid word indices
        """
        valid_actions = []
        for word in self.candidate_set:
            if word in self.word_to_idx:
                valid_actions.append(self.word_to_idx[word])
        return valid_actions
    
    def choose_word(self) -> Optional[str]:
        """
        Choose next word using epsilon-greedy policy with Q-network.
        
        Returns:
            Selected word or None if max guesses reached
        """
        if self.guess_number >= self.MAX_GUESSES:
            if self.verbose:
                print("Max guesses reached!")
            return None
        
        # Encode current state
        self.current_state = self._encode_state()
        
        # Get valid actions
        valid_actions = self._get_valid_actions()
        
        if not valid_actions:
            if self.verbose:
                print("No valid words remaining!")
            return None
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: random valid action
            action_idx = random.choice(valid_actions)
        else:
            # Exploitation: use Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
                
                # Mask invalid actions
                masked_q = np.full(self.action_dim, -np.inf)
                masked_q[valid_actions] = q_values[valid_actions]
                
                action_idx = np.argmax(masked_q)
        
        # Convert action to word
        guess = self.idx_to_word[action_idx]
        
        self.guess_number += 1
        self.guesses.append(guess)
        
        if self.verbose:
            print(f"Guess {self.guess_number}: {guess}")
        
        return guess
    
    def incorporate_guess_feedback(self, clue: list, state: int):
        """
        Incorporate feedback and update candidate set.
        Also stores experience for training.
        
        Args:
            clue: Feedback for the guess (0=gray, 1=yellow, 2=green)
            state: Game state (0=ongoing, 1=won, 2=lost)
        """
        super().incorporate_guess_feedback(clue, state)
        
        # Calculate reward
        reward = self._calculate_reward(clue, state)
        self.episode_reward += reward
        
        # Update candidate set based on clue
        self._update_candidates(clue, self.guesses[-1])
        
        # Get next state
        next_state = self._encode_state()
        done = (state != 0)
        
        # Store experience
        if self.current_state is not None and self.replay_buffer is not None:
            action_idx = self.word_to_idx[self.guesses[-1]]
            experience = Experience(
                self.current_state,
                action_idx,
                reward,
                next_state,
                done
            )
            self.replay_buffer.push(experience)
        
        # Train if enough experiences
        if self.replay_buffer is not None and len(self.replay_buffer) >= self.batch_size:
            self._train_step()
        
        if done:
            self._end_episode()
    
    def _calculate_reward(self, clue: list, state: int) -> float:
        """
        Calculate reward for the current guess.
        
        Args:
            clue: Feedback clue
            state: Game state
            
        Returns:
            Reward value
        """
        if state == 1:  # Won
            # Bonus for winning, scaled by efficiency
            remaining_guesses = self.MAX_GUESSES - self.guess_number
            return 5.0 + (remaining_guesses * 3.0)
        elif state == 2:  # Lost
            return -10.0
        else:  # Ongoing
            # Reward based on information gained
            green_count = sum(1 for c in clue if c == 2)
            yellow_count = sum(1 for c in clue if c == 1)
            
            # Reward for correct positions and correct letters
            reward = green_count * 1.0 + yellow_count * 0.5
            
            # Small penalty for each guess to encourage efficiency
            reward -= 0.1
            
            # Bonus for reducing candidate set significantly
            old_candidate_count = len(self.candidate_set)
            # (Candidate set will be updated after this, so we calculate expected reduction)
            
            return reward
    
    def _update_candidates(self, clue: list, guess: str):
        """
        Update candidate set based on clue feedback.
        
        Args:
            clue: Feedback for the guess
            guess: The guessed word
        """
        new_candidates = []
        
        for candidate in self.candidate_set:
            if self._is_valid_candidate(candidate, guess, clue):
                new_candidates.append(candidate)
        
        self.candidate_set = new_candidates
    
    def _is_valid_candidate(self, candidate: str, guess: str, clue: list) -> bool:
        """
        Check if a candidate word is consistent with the clue.
        
        Args:
            candidate: Candidate word to check
            guess: The guessed word
            clue: Feedback clue
            
        Returns:
            True if candidate is valid given the clue
        """
        # For each position in the guess
        for i, (guess_char, clue_val) in enumerate(zip(guess, clue)):
            if clue_val == 2:  # Green - must match at this position
                if candidate[i] != guess_char:
                    return False
            elif clue_val == 1:  # Yellow - letter exists but not at this position
                if candidate[i] == guess_char:
                    return False
                if guess_char not in candidate:
                    return False
            elif clue_val == 0:  # Gray - letter doesn't exist (with caveats)
                # Check if this letter appeared as green or yellow elsewhere
                has_green_or_yellow = any(
                    clue[j] > 0 and guess[j] == guess_char 
                    for j in range(len(guess))
                )
                if not has_green_or_yellow:
                    if guess_char in candidate:
                        return False
        
        return True
    
    def _train_step(self):
        """Perform one step of Q-learning training."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in batch])).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Track loss
        self.losses.append(loss.item())
    
    def _end_episode(self):
        """End of episode cleanup and tracking."""
        self.episode_rewards.append(self.episode_reward)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if self.verbose:
            print(f"Episode ended. Total reward: {self.episode_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        # Reset episode reward
        self.episode_reward = 0
    
    def save_model(self, path: str):
        """
        Save the Q-network model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, path)
        
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a pre-trained Q-network model.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
        
        if self.verbose:
            print(f"Model loaded from {path}")
    
    def get_training_stats(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        return {
            'episodes': len(self.episode_rewards),
            'total_steps': self.training_steps,
            'epsilon': self.epsilon,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss_last_100': np.mean(self.losses[-100:]) if self.losses else 0,
            'buffer_size': len(self.replay_buffer)
        }
