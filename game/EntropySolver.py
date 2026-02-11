# %%
import os, sys
from game.wordle import Wordle
from game.util import read_to_lines
from game.baseSolver import Solver
from game.constants import DEFAULT_GAME_CONFIG
from typing import Dict, List, Tuple, Set
import itertools as it
import numpy as np
import csv
import json
from scipy.special import expit
from scipy.stats import entropy
from tqdm import tqdm as ProgressDisplay
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


class EntropySolver(Solver):
    # TODO: 3b1b stores results as ternary ints instead of list of ints. Incorporate that into other code
    # Note: possible -> guess_set, allowed -> candidate_set

    def __init__(self, config: Dict[str, str] = DEFAULT_GAME_CONFIG, 
                first_guess = None, priors = None,
                manual = False, verbose=True, 
                CHUNK_LENGTH = 13000):
        super().__init__(config, manual, verbose)
        self.DATA_DIR = os.path.join(
            os.path.dirname(os.path.abspath('')),
            "data",
        )
        self.PATTERN_MATRIX_FILE = os.path.join(self.DATA_DIR, "pattern_matrix.npy")
        self.WORD_FREQ_FILE = os.path.join(self.DATA_DIR, "de_leipzig_frequencies.csv")
        self.WORD_FREQ_MAP_FILE = os.path.join(self.DATA_DIR, "freq_map.json")
        self.PATTERN_GRID_DATA = dict()
        self.CHUNK_LENGTH = CHUNK_LENGTH
        self.MISS = np.uint8(0)
        self.MISPLACED = np.uint8(1)
        self.EXACT = np.uint8(2)
        self.MAX_GUESSES = int(config['max_guesses'])
        self.patterns = []
        self.first_guess = first_guess   
        self.priors = priors  
        self.guess_number = 0     

    def get_uniform_prior(self):
        return dict(
            (w, 1)
            for w in self.guess_set
        )
    
    def get_true_wordle_prior(self):
        # TODO: so far guess set and candidate set are equal; make better
        return dict(
            (w, int(w in self.candidate_set))
            for w in self.guess_set
        )
    
    #TODO: improve this with hyperparameter tuning?
    def get_frequency_based_priors(self, n_common=3000, width_under_sigmoid=10):
        """
        We know that that list of wordle answers was curated by some human
        based on whether they're sufficiently common. This function aims
        to associate each word with the likelihood that it would actually
        be selected for the final answer.

        Sort the words by frequency, then apply a sigmoid along it.
        """
        freq_map = self.get_word_frequencies()
        words = np.array(list(freq_map.keys()))
        freqs = np.array([freq_map[w] for w in words])
        arg_sort = freqs.argsort()
        sorted_words = words[arg_sort]

        # We want to imagine taking this sorted list, and putting it on a number
        # line so that it's length is 10, situating it so that the n_common most common
        # words are positive, then applying a sigmoid
        x_width = width_under_sigmoid
        c = x_width * (-0.5 + n_common / len(words))
        xs = np.linspace(c - x_width / 2, c + x_width / 2, len(words))
        priors = dict()
        for word, x in zip(sorted_words, xs):
            priors[word] = expit(x)
        return priors

    def get_word_frequencies(self, regenerate=False):
        if os.path.exists(self.WORD_FREQ_MAP_FILE) or regenerate:
            with open(self.WORD_FREQ_MAP_FILE) as fp:
                result = json.load(fp)
            return result
        # Otherwise, regenerate
        freq_map = dict()
        with open('../data/de_leipzig_frequencies.csv', newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                freq_map[row["word"]] = float(row["rel_freq"])

        with open(self.WORD_FREQ_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(freq_map, f, indent=2, ensure_ascii=False)

        return freq_map

    def words_to_int_arrays(self, words):
        return np.array([[ord(c)for c in w] for w in words], dtype=np.uint8)
    
    def pattern_to_int_list(self, pattern):
        result = []
        curr = pattern
        for x in range(5):
            result.append(curr % 3)
            curr = curr // 3
        return result

    def int_list_to_pattern(self, digits):
        result = 0
        for i in range(5):
            result += digits[i] * (3 ** i)
        return result

    def pattern_to_string(self, pattern):
        d = {self.MISS: "⬛", self.MISPLACED: "🟨", self.EXACT: "🟩"}
        return "".join(d[x] for x in self.pattern_to_int_list(pattern))


    def patterns_to_string(self, patterns):
        return "\n".join(map(self.pattern_to_string, patterns))


    def generate_pattern_matrix(self, words1, words2):
        """
        A pattern for two words represents the wordle-similarity
        pattern (grey -> 0, yellow -> 1, green -> 2) but as an integer
        between 0 and 3^5. Reading this integer in ternary gives the
        associated pattern.

        This function computes the pairwise patterns between two lists
        of words, returning the result as a grid of hash values. Since
        this can be time-consuming, many operations that can be are vectorized
        (perhaps at the expense of easier readibility), and the the result
        is saved to file so that this only needs to be evaluated once, and
        all remaining pattern matching is a lookup.
        """

        # Number of letters/words
        nl = len(words1[0])
        nw1 = len(words1)  # Number of words
        nw2 = len(words2)  # Number of words

        # Convert word lists to integer arrays
        word_arr1, word_arr2 = map(self.words_to_int_arrays, (words1, words2))

        # equality_grid keeps track of all equalities between all pairs
        # of letters in words. Specifically, equality_grid[a, b, i, j]
        # is true when words[i][a] == words[b][j]
        equality_grid = np.zeros((nw1, nw2, nl, nl), dtype=bool)
        for i, j in it.product(range(nl), range(nl)):
            equality_grid[:, :, i, j] = np.equal.outer(word_arr1[:, i], word_arr2[:, j])

        # full_pattern_matrix[a, b] should represent the 5-color pattern
        # for guess a and answer b, with 0 -> grey, 1 -> yellow, 2 -> green
        full_pattern_matrix = np.zeros((nw1, nw2, nl), dtype=np.uint8)

        # Green pass
        for i in range(nl):
            matches = equality_grid[:, :, i, i].flatten()  # matches[a, b] is true when words[a][i] = words[b][i]
            full_pattern_matrix[:, :, i].flat[matches] = self.EXACT

            for k in range(nl):
                # If it's a match, mark all elements associated with
                # that letter, both from the guess and answer, as covered.
                # That way, it won't trigger the yellow pass.
                equality_grid[:, :, k, i].flat[matches] = False
                equality_grid[:, :, i, k].flat[matches] = False

        # Yellow pass
        for i, j in it.product(range(nl), range(nl)):
            matches = equality_grid[:, :, i, j].flatten()
            full_pattern_matrix[:, :, i].flat[matches] = self.MISPLACED
            for k in range(nl):
                # Similar to above, we want to mark this letter
                # as taken care of, both for answer and guess
                equality_grid[:, :, k, j].flat[matches] = False
                equality_grid[:, :, i, k].flat[matches] = False

        # Rather than representing a color pattern as a lists of integers,
        # store it as a single integer, whose ternary representations corresponds
        # to that list of integers.
        pattern_matrix = np.dot(
            full_pattern_matrix,
            (3**np.arange(nl)).astype(np.uint8)
        )

        return pattern_matrix

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def generate_pattern_matrix_in_blocks(self, many_words1, many_words2):
        block_matrix = None
        for words1 in self.chunks(many_words1, self.CHUNK_LENGTH):
            row = None

            for words2 in self.chunks(many_words2, self.CHUNK_LENGTH):
                block = self.generate_pattern_matrix(words1, words2)

                if row is None:
                    row = block
                else:
                    row = np.hstack((row, block))

            if block_matrix is None:
                block_matrix = row
            else:
                block_matrix = np.vstack((block_matrix, row))

        return block_matrix

    def generate_full_pattern_matrix(self):
        pattern_matrix = self.generate_pattern_matrix_in_blocks(self.guess_set, self.guess_set)
        # Save to file
        np.save(self.PATTERN_MATRIX_FILE, pattern_matrix)
        return pattern_matrix

    def get_pattern_matrix(self, words1, words2):
        if not self.PATTERN_GRID_DATA:
            if not os.path.exists(self.PATTERN_MATRIX_FILE):
                log.info("\n".join([
                    "Generating pattern matrix. This takes a minute, but",
                    "the result will be saved to file so that it only",
                    "needs to be computed once.",
                ]))
                self.generate_full_pattern_matrix()
            self.PATTERN_GRID_DATA['grid'] = np.load(self.PATTERN_MATRIX_FILE)
            self.PATTERN_GRID_DATA['words_to_index'] = dict(zip(
                self.guess_set, it.count()
            ))

        full_grid = self.PATTERN_GRID_DATA['grid']
        words_to_index = self.PATTERN_GRID_DATA['words_to_index']

        indices1 = [words_to_index[w] for w in words1]
        indices2 = [words_to_index[w] for w in words2]

        return full_grid[np.ix_(indices1, indices2)]

    def get_pattern(self, guess, answer):
        if self.PATTERN_GRID_DATA:
            saved_words = self.PATTERN_GRID_DATA['words_to_index']
            if guess in saved_words and answer in saved_words:
                return self.get_pattern_matrix([guess], [answer])[0, 0]
        return self.generate_pattern_matrix([guess], [answer])[0, 0]

    def get_possible_words(self, guess, pattern, word_list):
        all_patterns = self.get_pattern_matrix([guess], word_list).flatten()
        return list(np.array(word_list)[all_patterns == pattern])
    
    def get_word_buckets(self, guess):
        buckets = [[] for x in range(3**5)] #TODO: is this **5 because of word length? If so, replace by self.N
        hashes = self.get_pattern_matrix([guess], self.guess_set).flatten()
        for index, word in zip(hashes, self.guess_set):
            buckets[index].append(word)
        return buckets

    # Entropy calculation starts here

    def get_weights(self, words, priors):
        frequencies = np.array([priors[word] for word in words])
        total = frequencies.sum()
        if total == 0:
            return np.zeros(frequencies.shape)
        return frequencies / total


    def get_pattern_distributions(self, allowed_words, possible_words, weights):
        """
        For each possible guess in allowed_words, this finds the probability
        distribution across all of the 3^5 wordle patterns you could see, assuming
        the possible answers are in possible_words with associated probabilities
        in weights.

        It considers the pattern hash grid between the two lists of words, and uses
        that to bucket together words from possible_words which would produce
        the same pattern, adding together their corresponding probabilities.
        """
        pattern_matrix = self.get_pattern_matrix(allowed_words, possible_words)

        n = len(allowed_words)
        distributions = np.zeros((n, 3**5))
        n_range = np.arange(n)
        for j, prob in enumerate(weights):
            distributions[n_range, pattern_matrix[:, j]] += prob
        return distributions


    def entropy_of_distributions(self, distributions, atol=1e-12):
        axis = len(distributions.shape) - 1
        return entropy(distributions, base=2, axis=axis)


    def get_entropies(self, cands, guesses, weights):
        if weights.sum() == 0:
            return np.zeros(len(cands))
        distributions = self.get_pattern_distributions(cands, guesses, weights)
        return self.entropy_of_distributions(distributions)


    def max_bucket_size(self, guess, possible_words, weights):
        dist = self.get_pattern_distributions([guess], possible_words, weights)
        return dist.max()


    def words_to_max_buckets(self, possible_words, weights):
        return dict(
            (word, self.max_bucket_size(word, possible_words, weights))
            for word in ProgressDisplay(possible_words)
        )
        # TODO: rudiment? what use in 3b1b
        """
        words_and_maxes = list(w2m.items())
        words_and_maxes.sort(key=lambda t: t[1])
        words_and_maxes[:-20:-1]
        """


    def get_bucket_sizes(self, allowed_words, possible_words):
        """
        Returns a (len(allowed_words), 243) shape array reprenting the size of
        word buckets associated with each guess in allowed_words
        """
        weights = np.ones(len(possible_words))
        return self.get_pattern_distributions(allowed_words, possible_words, weights)


    def get_bucket_counts(self, allowed_words, possible_words):
        """
        Returns the number of separate buckets that each guess in allowed_words
        would separate possible_words into
        """
        bucket_sizes = self.get_bucket_sizes(allowed_words, possible_words)
        return (bucket_sizes > 0).sum(1)
        
    # Solvers
    def get_guess_values_array(self, priors, look_two_ahead=False):
        weights = self.get_weights(self.guess_set, priors)
        ents1 = self.get_entropies(self.candidate_set, self.guess_set, weights)
        probs = np.array([
            0 if word not in self.guess_set else weights[self.guess_set.index(word)]
            for word in self.candidate_set
        ])

        return np.array([ents1, probs])

    def entropy_to_expected_score(self, ent):
        """
        Based on a regression associating entropies with typical scores
        from that point forward in simulated games, this function returns
        what the expected number of guesses required will be in a game where
        there's a given amount of entropy in the remaining possibilities.
        """
        # Assuming you can definitely get it in the next guess,
        # this is the expected score
        min_score = 2**(-ent) + 2 * (1 - 2**(-ent))

        # To account for the likely uncertainty after the next guess,
        # and knowing that entropy of 11.5 bits seems to have average
        # score of 3.5, we add a line to account
        # we add a line which connects (0, 0) to (3.5, 11.5)
        return min_score + 1.5 * ent / 11.5


    def get_expected_scores(self, allowed_words, possible_words, priors,
                            look_two_ahead=False,
                            n_top_candidates_for_two_step=25,
                            ):
        # Currenty entropy of distribution
        weights = self.get_weights(possible_words, priors)
        H0 = self.entropy_of_distributions(weights)
        H1s = self.get_entropies(allowed_words, possible_words, weights)

        word_to_weight = dict(zip(possible_words, weights))
        probs = np.array([word_to_weight.get(w, 0) for w in allowed_words])
        # If this guess is the true answer, score is 1. Otherwise, it's 1 plus
        # the expected number of guesses it will take after getting the corresponding
        # amount of information.
        expected_scores = probs + (1 - probs) * (1 + self.entropy_to_expected_score(H0 - H1s))

        if not look_two_ahead:
            return expected_scores

        # For the top candidates, refine the score by looking two steps out
        # This is currently quite slow, and could be optimized to be faster.
        # But why?
        sorted_indices = np.argsort(expected_scores)
        allowed_second_guesses = self.guess_set
        expected_scores += 1  # Push up the rest
        for i in ProgressDisplay(sorted_indices[:n_top_candidates_for_two_step], leave=False):
            guess = allowed_words[i]
            H1 = H1s[i]
            dist = self.get_pattern_distributions([guess], possible_words, weights)[0]
            buckets = self.get_word_buckets(guess, possible_words)
            second_guesses = [
                self.optimal_guess(allowed_second_guesses, bucket, priors, look_two_ahead=False)
                for bucket in buckets
            ]
            H2s = [
                self.get_entropies([guess2], bucket, get_weights(bucket, priors))[0]
                for guess2, bucket in zip(second_guesses, buckets)
            ]

            prob = word_to_weight.get(guess, 0)
            expected_scores[i] = sum((
                # 1 times Probability guess1 is correct
                1 * prob,
                # 2 times probability guess2 is correct
                2 * (1 - prob) * sum(
                    p * word_to_weight.get(g2, 0)
                    for p, g2 in zip(dist, second_guesses)
                ),
                # 2 plus expected score two steps from now
                (1 - prob) * (2 + sum(
                    p * (1 - word_to_weight.get(g2, 0)) * entropy_to_expected_score(H0 - H1 - H2)
                    for p, g2, H2 in zip(dist, second_guesses, H2s)
                ))
            ))
        return expected_scores


    def optimal_guess(self, 
                    allowed_words = None,
                    possible_words = None,
                    priors = None,
                    look_two_ahead=False,
                    optimize_for_uniform_distribution=False,
                    purely_maximize_information=False,
                    ):
        if allowed_words is None:
            allowed_words = self.guess_set
        if possible_words is None:
            possible_words = self.candidate_set
        if priors is None:
            priors = self.get_frequency_based_priors()
        

        if purely_maximize_information:
            if len(possible_words) == 1:
                return possible_words[0]
            weights = self.get_weights(possible_words, priors)
            ents = self.get_entropies(allowed_words, possible_words, weights)
            return allowed_words[np.argmax(ents)]

        # Just experimenting here...
        if optimize_for_uniform_distribution:
            expected_scores = self.get_score_lower_bounds(
                allowed_words, possible_words
            )
        else:
            expected_scores = self.get_expected_scores(
                allowed_words, possible_words, priors,
                look_two_ahead=look_two_ahead
            )
        return allowed_words[np.argmin(expected_scores)]


    def brute_force_optimal_guess(self, all_words, possible_words, priors, n_top_picks=10, display_progress=False):
        if len(possible_words) == 0:
            # Doesn't matter what to return in this case, so just default to first word in list.
            return all_words[0]
        # For the suggestions with the top expected scores, just
        # actually play the game out from this point to see what
        # their actual scores are, and minimize.
        expected_scores = self.get_score_lower_bounds(all_words, possible_words)
        top_choices = [all_words[i] for i in np.argsort(expected_scores)[:n_top_picks]]
        true_average_scores = []
        if display_progress:
            iterable = ProgressDisplay(
                top_choices,
                desc=f"Possibilities: {len(possible_words)}",
                leave=False
            )
        else:
            iterable = top_choices

        for next_guess in iterable:
            scores = []
            for answer in possible_words:
                score = 1
                possibilities = list(possible_words)
                guess = next_guess
                while guess != answer:
                    possibilities = self.get_possible_words(
                        guess, self.get_pattern(guess, answer),
                        possibilities,
                    )
                    # Make recursive? If so, we'd want to keep track of
                    # the next_guess map and pass it down in the recursive
                    # subcalls
                    guess = self.optimal_guess(
                        all_words, possibilities, priors,
                        optimize_for_uniform_distribution=True
                    )
                    score += 1
                scores.append(score)
            true_average_scores.append(np.mean(scores))
        return top_choices[np.argmin(true_average_scores)]

    # TODO: adapt this for my purposes
    def gather_entropy_to_score_data(self, first_guess="raine", priors=None):
        words = self.guess_set
        answers = self.candidate_set
        if priors is None:
            priors = self.get_true_wordle_prior()

        # List of entropy/score pairs
        ent_score_pairs = []

        for answer in ProgressDisplay(answers):
            score = 1
            possibilities = list(filter(lambda w: priors[w] > 0, words))
            guess = first_guess
            guesses = []
            entropies = []
            while True:
                guesses.append(guess)
                weights = self.get_weights(possibilities, priors)
                entropies.append(self.entropy_of_distributions(weights))
                if guess == answer:
                    break
                possibilities = self.get_possible_words(
                    guess, self.get_pattern(guess, answer), possibilities
                )
                guess = self.optimal_guess(words, possibilities, priors)
                score += 1

            for sc, ent in zip(it.count(1), reversed(entropies)):
                ent_score_pairs.append((ent, sc))

        with open(ENT_SCORE_PAIRS_FILE, 'w') as fp:
            json.dump(ent_score_pairs, fp)

        return ent_score_pairs


    def simulate_games(self, first_guess=None,
                    priors=None,
                    look_two_ahead=False,
                    optimize_for_uniform_distribution=False,
                    second_guess_map=None,
                    exclude_seen_words=False,
                    test_set=None,
                    shuffle=False,
                    hard_mode=False,
                    purely_maximize_information=False,
                    brute_force_optimize=False,
                    brute_force_depth=10,
                    results_file=None,
                    next_guess_map_file=None,
                    quiet=False,
                    ):
        all_words = self.guess_set
        short_word_list = self.candidate_set

        if priors is None:
            priors = self.get_frequency_based_priors()
        
        if first_guess is None:
            first_guess = self.optimal_guess(
                all_words, all_words, priors,
                look_two_ahead,
                optimize_for_uniform_distribution,
                purely_maximize_information,
            )

        if test_set is None:
            test_set = short_word_list

        if shuffle:
            random.shuffle(test_set)

        seen = set()

        # Function for choosing the next guess, with a dict to cache
        # and reuse results that are seen multiple times in the sim
        next_guess_map = {}

        def get_next_guess(guesses, patterns, possibilities):
            phash = "".join(
                str(g) + "".join(map(str, self.pattern_to_int_list(p)))
                for g, p in zip(guesses, patterns)
            )
            if second_guess_map is not None and len(patterns) == 1:
                next_guess_map[phash] = second_guess_map[patterns[0]]
            if phash not in next_guess_map:
                choices = all_words
                if hard_mode:
                    for guess, pattern in zip(guesses, patterns):
                        choices = self.get_possible_words(guess, pattern, choices)
                if brute_force_optimize:
                    next_guess_map[phash] = self.brute_force_optimal_guess(
                        choices, possibilities, priors,
                        n_top_picks=brute_force_depth,
                    )
                else:
                    next_guess_map[phash] = self.optimal_guess(
                        choices, possibilities, priors,
                        look_two_ahead=look_two_ahead,
                        purely_maximize_information=purely_maximize_information,
                        optimize_for_uniform_distribution=optimize_for_uniform_distribution,
                    )
            return next_guess_map[phash]

        # Go through each answer in the test set, play the game,
        # and keep track of the stats.
        scores = np.zeros(0, dtype=int)
        game_results = []
        for answer in ProgressDisplay(test_set, leave=False, desc=" Trying all wordle answers"):
            guesses = []
            patterns = []
            possibility_counts = []
            possibilities = list(filter(lambda w: priors[w] > 0, all_words))

            if exclude_seen_words:
                possibilities = list(filter(lambda w: w not in seen, possibilities))

            score = 1
            guess = first_guess
            while guess != answer:
                pattern = self.get_pattern(guess, answer)
                guesses.append(guess)
                patterns.append(pattern)
                possibilities = self.get_possible_words(guess, pattern, possibilities)
                possibility_counts.append(len(possibilities))
                score += 1
                guess = get_next_guess(guesses, patterns, possibilities)

            # Accumulate stats
            scores = np.append(scores, [score])
            score_dist = [
                int((scores == i).sum())
                for i in range(1, scores.max() + 1)
            ]
            total_guesses = scores.sum()
            average = scores.mean()
            seen.add(answer)

            game_results.append(dict(
                score=int(score),
                answer=answer,
                guesses=guesses,
                patterns=list(map(int, patterns)),
                reductions=possibility_counts,
            ))
            # Print outcome
            if not quiet:
                message = "\n".join([
                    "",
                    f"Score: {score}",
                    f"Answer: {answer}",
                    f"Guesses: {guesses}",
                    f"Reductions: {possibility_counts}",
                    *self.patterns_to_string((*patterns, 3**5 - 1)).split("\n"),
                    *" " * (6 - len(patterns)),
                    f"Distribution: {score_dist}",
                    f"Total guesses: {total_guesses}",
                    f"Average: {average}",
                    *" " * 2,
                ])
                if answer is not test_set[0]:
                    # Move cursor back up to the top of the message
                    n = len(message.split("\n")) + 1
                    print(("\033[F\033[K") * n)
                else:
                    print("\r\033[K\n")
                print(message)

        final_result = dict(
            score_distribution=score_dist,
            total_guesses=int(total_guesses),
            average_score=float(scores.mean()),
            game_results=game_results,
        )

        # Save results
        for obj, file in [(final_result, results_file), (next_guess_map, next_guess_map_file)]:
            if file:
                path = os.path.join(DATA_DIR, "simulation_results", file)
                with open(path, 'w') as fp:
                    json.dump(obj, fp)

        return final_result, next_guess_map

    #TODO: find way to reuse common guesses?
    def get_next_guess(self, possibilities, priors,
                        next_guess_map = {}, second_guess_map = None,
                        hard_mode = False, brute_force_optimize = False,
                        look_two_ahead = False, purely_maximize_information = False,
                        optimize_for_uniform_distribution = False
                        ):
            # TODO: add init if first guess; where do best?; TODO: add priors as self.
            if self.guess_number == 0:
                if self.first_guess is None:
                    self.first_guess = self.optimal_guess(
                        self.candidate_set, self.candidate_set, self.priors,
                        look_two_ahead=False,
                        optimize_for_uniform_distribution=False,
                        purely_maximize_information=False,
                        )
                self.guesses.append(self.first_guess)
                self.guess_number += 1
                return self.first_guess
            
            else:
                phash = "".join(
                    str(g) + "".join(map(str, self.pattern_to_int_list(p)))
                    for g, p in zip(self.guesses, self.patterns)
                )
                if second_guess_map is not None and len(self.patterns) == 1:
                    next_guess_map[phash] = second_guess_map[self.patterns[0]]
                if phash not in next_guess_map:
                    choices = self.guess_set
                    if hard_mode:
                        for guess, pattern in zip(self.guesses, self.patterns):
                            choices = self.get_possible_words(guess, pattern, choices)
                    if brute_force_optimize:
                        next_guess_map[phash] = self.brute_force_optimal_guess(
                            choices, possibilities, priors,
                            n_top_picks=brute_force_depth,
                        )
                    else:
                        next_guess_map[phash] = self.optimal_guess(
                            choices, possibilities, priors,
                            look_two_ahead=look_two_ahead,
                            purely_maximize_information=purely_maximize_information,
                            optimize_for_uniform_distribution=optimize_for_uniform_distribution,
                        )
                self.guesses.append(next_guess_map[phash])
                self.guess_number += 1
                return next_guess_map[phash]
        
    def incorporate_clue(self, int_list, solution_state):
        pat = self.int_list_to_pattern(int_list)
        self.patterns.append(pat)
        self.states.append(solution_state)

    def one_round_loop(self, int_list, solution_state):
        pat = self.int_list_to_pattern(int_list)
        self.patterns.append(pat)
        self.states.append(solution_state)
        last_guess = self.guesses[-1]
        self.possibilities = self.get_possible_words(last_guess, pat, self.possibilities)
        guess = self.get_next_guess(self.possibilities, self.priors)
        return(guess)

    def round_zero(self):
        if self.priors is None:
            self.priors = self.get_frequency_based_priors()
        
        self.possibilities = list(filter(lambda w: self.priors[w] > 0, self.guess_set))

        guess = self.get_next_guess(self.possibilities, self.priors)

        return(guess)



    def play_wordle(self, 
                    answer, 
                    priors, 
                    first_guess="raine", 
                    look_two_ahead=False,
                    optimize_for_uniform_distribution=False,
                    purely_maximize_information=False,
                    quiet=False,
                    ):
        guesses = []
        patterns = []
        possibility_counts = []
        all_words = self.guess_set
        
        if priors is None:
            priors = self.get_frequency_based_priors()
        
        if first_guess is None:
            first_guess = self.optimal_guess(
                all_words, all_words, priors,
                look_two_ahead,
                optimize_for_uniform_distribution,
                purely_maximize_information,
                )

        possibilities = list(filter(lambda w: priors[w] > 0, all_words))

        score = 1
        guess = first_guess
        while guess != answer:
            pattern = self.get_pattern(guess, answer)
            guesses.append(guess)
            patterns.append(pattern)
            possibilities = self.get_possible_words(guess, pattern, possibilities)
            possibility_counts.append(len(possibilities))
            score += 1
            guess = self.get_next_guess(guesses, patterns, possibilities, priors)

        # Print outcome
        if not quiet:
            message = "\n".join([
                "",
                f"Score: {score}",
                f"Answer: {answer}",
                f"Guesses: {guesses}",
                f"Reductions: {possibility_counts}",
                *self.patterns_to_string((*patterns, 3**5 - 1)).split("\n"),
                *" " * (6 - len(patterns)),
                f"Distribution: {score_dist}",
                f"Total guesses: {total_guesses}",
                f"Average: {average}",
                *" " * 2,
            ])
            if answer is not test_set[0]:
                # Move cursor back up to the top of the message
                n = len(message.split("\n")) + 1
                print(("\033[F\033[K") * n)
            else:
                print("\r\033[K\n")
            print(message)
        
        return dict(
            score=int(score),
            answer=answer,
            guesses=guesses,
            patterns=list(map(int, patterns)),
            reductions=possibility_counts,
        )

    

    def multithread_evaluate(self, 
                first_guess=None,
                priors=None,
                look_two_ahead=False,
                optimize_for_uniform_distribution=False,
                second_guess_map=None,
                exclude_seen_words=False,
                test_set=None,
                shuffle=False,
                hard_mode=False,
                purely_maximize_information=False,
                brute_force_optimize=False,
                brute_force_depth=10,
                results_file=None,
                next_guess_map_file=None,
                quiet=False,
                ):
        
        all_words = self.guess_set
        short_word_list = self.candidate_set

        if priors is None:
            priors = self.get_frequency_based_priors()
        
        if first_guess is None:
            first_guess = self.optimal_guess(
                all_words, all_words, priors,
                look_two_ahead,
                optimize_for_uniform_distribution,
                purely_maximize_information,
            )

        if test_set is None:
            test_set = short_word_list

        if shuffle:
            random.shuffle(test_set)

        seen = set()

        # Function for choosing the next guess, with a dict to cache
        # and reuse results that are seen multiple times in the sim
        next_guess_map = {}
        scores = np.zeros(0, dtype=int)
        game_results = []

        with ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            results = list(ProgressDisplay(executor.map(
            self.play_wordle,
            test_set, 
            repeat(priors), 
            repeat(first_guess), 
            repeat(look_two_ahead),                   
            repeat(optimize_for_uniform_distribution),
            repeat(purely_maximize_information),
            repeat(quiet)), 
            total=len(test_set)))

        return results
# %%
