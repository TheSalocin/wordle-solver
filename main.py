# %%
import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: 
    sys.path.append(dir1)

from game.wordle import Wordle
from game.util import read_to_lines
from game.EntropySolver import EntropySolver
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import argparse


# %%
guess_words = read_to_lines("data/de_leipzig_5_letter.txt")
answer_words = read_to_lines("data/de_wiktionary_5_letter_shortlist.txt")

de_config = {
    'max_guesses': str(6),
    # The set of words that can potentially be solutions
    'candidate_set': answer_words,
    # The set of words that can be guessed validly
    'guess_set': guess_words
}

def run_wordle(answer, first_guess, verbose):
    w = Wordle(answer, config = de_config, verbose = verbose)
    solve = EntropySolver(config=de_config, first_guess=first_guess)
    guess = solve.round_zero()
    state = 0
    while state == 0:
        int_list, state = w.guess(guess)
        guess = solve.one_round_loop(int_list, state)
    res = solve.guess_number
    del solve
    del w
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate solver')
    parser.add_argument('-s',
                        '--solver',
                        help='Solver to use. Default Entropy',
                        choices=["Entropy"],
                        default="Entropy",
                        required=True)
    parser.add_argument('-k',
                        type=int,
                        help='Number of candidates to eval. if none, all words are evaluated',
                        default=None,
                        required=False)
    parser.add_argument('-g',
                        '--generate',
                        help='Whether to generate pattern matrix for EntropySolver',
                        default=False,
                        required=False)
    parser.add_argument('-f',
                        '--first_guess',
                        help='first word to guess',
                        default=None,
                        required=False)
    parser.add_argument('-v',
                        '--verbose',
                        help='whether to run in verbose mode',
                        default=False,
                        required=False)
    args = parser.parse_args()
    
    if args.k is not None:
        random.seed(2026)
        test_words = random.choices(answer_words, k = args.k)
    else:
        test_words = answer_words
    


    n_cores = os.cpu_count()
    max_workers = int(n_cores // 2) 

    n_guesses = []
    if args.generate and (args.solver == "Entropy"):
        print("generating full pattern matrix")
        es = EntropySolver(config = de_config, first_guess = args.first_guess)
        es.generate_full_pattern_matrix()
    

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_wordle, answer, args.first_guess, args.verbose) for answer in test_words]

            for future in tqdm(as_completed(futures), total=len(futures)):
                n_guesses.append(future.result())
                
    # 🔥 All jobs finished here
    print("All jobs completed. Saving results...")

    with open(f"results_{args.first_guess}.txt", "w", encoding="utf-8") as f:
        for item in n_guesses:
            f.write(str(item) + "\n")

    print("Saved successfully.")
# %%
