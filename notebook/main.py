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

# %%
guess_words = read_to_lines("../data/de_leipzig_5_letter.txt")
answer_words = read_to_lines("../data/de_wiktionary_5_letter_shortlist.txt")

de_config = {
    'max_guesses': str(300),
    # The set of words that can potentially be solutions
    'candidate_set': answer_words,
    # The set of words that can be guessed validly
    'guess_set': guess_words
}

def run_wordle(answer):
    w = Wordle(answer, config = de_config, verbose = False)
    solve = EntropySolver(config=de_config, first_guess="raine")
    guess = solve.round_zero()
    state = 0
    while state == 0:
        int_list, state = w.guess(guess)
        guess = solve.one_round_loop(int_list, state)
    return solve.guess_number

if __name__ == "__main__":
    n_cores = os.cpu_count()
    max_workers = n_cores - 1

    n_guesses = []


    with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_wordle, answer) for answer in answer_words]

            for future in tqdm(as_completed(futures), total=len(futures)):
                n_guesses.append(future.result())
                
    # 🔥 All jobs finished here
    print("All jobs completed. Saving results...")

    with open("results.json", "w") as f:
        json.dump(results, f)

    print("Saved successfully.")