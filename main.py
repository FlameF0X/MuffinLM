########################################################
#   Muffin V2.1 -- VERSION 3 (code name: Muffin+)      #
#   It is better at text generation+!                  #
########################################################

import random
import math
from collections import defaultdict
from typing import List, Dict, Tuple


class TextGenerator:
    def __init__(self, corpus_path: str, n: int = 3) -> None:
        self.n = n  # The n in n-gram (3 for trigrams)
        self.chain: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.corpus = self.load_corpus(corpus_path)
        self.train(self.corpus)

    def load_corpus(self, file_path: str) -> str:
        """Load the corpus from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_words(self, input_text: str) -> List[str]:
        """Split a string into words."""
        return input_text.split()

    def train(self, corpus: str) -> None:
        """Train the model by building an n-gram transition map from the corpus."""
        self.words = self.split_words(corpus)  # Store all words for random selection
        for i in range(len(self.words) - self.n + 1):
            current_ngram = tuple(self.words[i:i + self.n - 1])
            next_word = self.words[i + self.n - 1]
            self.chain[current_ngram][next_word] += 1

    def weighted_random(self, choices: Dict[str, int], temperature: float, k: int) -> str:
        """Generates a random word based on weighted probabilities."""
        total = sum(math.exp(count / temperature) for count in choices.values())
        probabilities = {word: (math.exp(count / temperature) / total) for word, count in choices.items()}

        # Sort words by probabilities and take top-k choices
        top_choices = sorted(probabilities, key=probabilities.get, reverse=True)[:k]

        # Select from top-k choices randomly
        return random.choice(top_choices)

    def generate(self, start_words: str, length: int, temperature: float, k: int) -> str:
        """Generates a sequence of words based on the N-Gram model."""
        current_ngram = tuple(start_words.split()[:self.n - 1])
        result = list(current_ngram)

        while len(result) < length:
            next_words = self.chain.get(current_ngram)
            if not next_words:
                break

            next_word = self.weighted_random(next_words, temperature, k)
            result.append(next_word)

            current_ngram = tuple(result[-(self.n - 1):])

            # Optional: Break if halfway through and ends with punctuation
            if len(result) > length // 2 and self.ends_with_punctuation(result[-1]):
                break

        return ' '.join(result)

    @staticmethod
    def ends_with_punctuation(word: str) -> bool:
        """Check if the last word ends with punctuation."""
        return word[-1] in {'.', '!', '?', ';', ':'}

    def get_random_starting_words(self, word_count: int = 2) -> str:
        """Select random starting words that exist in the corpus."""
        if len(self.words) < word_count:
            raise ValueError("Not enough words in the corpus for starting sequence.")

        start_index = random.randint(0, len(self.words) - word_count)
        return ' '.join(self.words[start_index:start_index + word_count])


# Use the larger corpus dataset (dataset-4.txt)
corpus_file_path = '/dataset-4.txt'

# Initialize the text generator with the trigram model (n=3)
generator = TextGenerator(corpus_file_path, n=3)

# Randomly select starting words from the dataset
start_words = generator.get_random_starting_words(word_count=3)
length = 20  # Length of the generated text
temperature = 0.835  # Adjust the randomness
k = 5  # Number of top choices to consider

# Generate text starting with the randomly selected start_words
generated_text = generator.generate(start_words, length, temperature, k)

print("Starting Words: " + start_words)
print("Generated Text: " + generated_text)
