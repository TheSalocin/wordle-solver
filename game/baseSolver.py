# game/baseSolver.py

import os, sys
from game.util import get_n_from_word_set
from game.constants import DEFAULT_GAME_CONFIG
from typing import Dict, List, Tuple, Set
import random

class Solver:
    
    def __init__(self, config: Dict[str, str] = DEFAULT_GAME_CONFIG, manual = False, verbose=True):
        
        if not 'candidate_set' in config or not len(config['candidate_set']): 
            raise Exception('candidate_set not specified in config')
        self.candidate_set = [w.lower() for w in set(config['candidate_set'])]
        
        if not 'guess_set' in config or not len(config['guess_set']):
            self.guess_set = [w.lower() for w in set(config['candidate_set'])]
        else:
            self.guess_set = [w.lower() for w in set(config['guess_set'])]
        self.N = get_n_from_word_set(config['candidate_set'])
        self.MAX_GUESSES = int(config['max_guesses'])
        self.guess_number = 0
        self.guesses = []
        self.clues = []
        self.states = []
        self.verbose = verbose
        self.manual = manual

    def choose_word(self):
        # random guesses for now to test
        if self.guess_number >= self.MAX_GUESSES:
            if self.verbose:
                print("Max guesses reached!")
            return None
        
        guess = random.choice(self.candidate_set)
        self.guess_number += 1
        self.guesses.append(guess)

        return guess
    
    def incorporate_guess_feedback(self, clue: list, state: int):
        self.clues.append(clue)
        self.states.append(state)
    
        
