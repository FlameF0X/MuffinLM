########################################################
#   Muffin V2 -- VERSION 2 (code name: Story Muffin)   #
#   It is more better at text generation!!             #
########################################################

import random
import math
from collections import defaultdict


class TextGenerator:
    def __init__(self, corpus_path, n=3):
        self.n = n  # The n in n-gram (3 for trigrams)
        self.chain = defaultdict(lambda: defaultdict(int))
        self.corpus = self.load_corpus(corpus_path)
        self.train(self.corpus)

    # Load the corpus from a file
    def load_corpus(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    # Splits a string into words
    def split_words(self, input_text):
        return input_text.split()

    # Trains the model by building an n-gram transition map from the corpus
    def train(self, corpus):
        self.words = self.split_words(corpus)  # Store all words for random selection
        for i in range(len(self.words) - self.n):
            # Create a tuple of the current n-1 words (history) and the next word (future)
            current_ngram = tuple(self.words[i:i + self.n - 1])
            next_word = self.words[i + self.n - 1]
            self.chain[current_ngram][next_word] += 1

    # Generates a random word based on a list of possible words, adjusted for temperature and top-k
    def weighted_random(self, choices, temperature, k):
        total = 0
        weights = {}

        # Calculate weights based on counts and temperature
        for word, count in choices.items():
            weights[word] = math.exp(count / temperature)  # Apply temperature
            total += weights[word]

        # Normalize weights and create a list of word probabilities
        probabilities = {word: weight / total for word, weight in weights.items()}

        # Sort words by probabilities and take top-k
        sorted_words = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

        # Limit to top-k choices
        top_choices = [word for word, _ in sorted_words[:k]]

        # Select from top-k choices randomly
        return random.choice(top_choices)

    # Generates a sequence of words based on the N-Gram model
    def generate(self, start_words, length, temperature, k):
        current_ngram = tuple(start_words.split()[:self.n - 1])
        result = list(current_ngram)

        while len(result) < length:
            next_words = self.chain[current_ngram]
            if not next_words:
                break

            # Pick a random next word based on the n-gram probabilities
            next_word = self.weighted_random(next_words, temperature, k)
            result.append(next_word)

            # Update the current n-gram (move forward in the sequence)
            current_ngram = tuple(result[-(self.n - 1):])

            # Optional: Allow for a longer sequence before checking punctuation
            if len(result) > length // 2 and self.ends_with_punctuation(result[-1]):
                break

        return ' '.join(result)

    # Check if the last word ends with punctuation
    def ends_with_punctuation(self, word):
        return word[-1] in {'.', '!', '?', ';', ':'}

    # Select random starting words that exist in the corpus
    def get_random_starting_words(self, word_count=2):
        if len(self.words) < word_count:
            raise ValueError("Not enough words in the corpus for starting sequence.")

        start_index = random.randint(0, len(self.words) - word_count)
        return ' '.join(self.words[start_index:start_index + word_count])


# Use the larger corpus dataset (dataset-3.txt)
corpus_file_path = '/dataset-3.txt'

# Initialize the text generator with the trigram model (n=3)
generator = TextGenerator(corpus_file_path, n=3)

# Randomly select starting words from the dataset
start_words = generator.get_random_starting_words(word_count=3)  # You can choose how many words to start with
length = 500  # Length of the generated text
temperature = 0.835  # Adjust the randomness
k = 5  # Number of top choices to consider

# Generate text starting with the randomly selected start_words
generated_text = generator.generate(start_words, length, temperature, k)

print("Starting Words: " + start_words)
print("Generated Text: " + generated_text)
