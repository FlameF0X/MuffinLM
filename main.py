########################################################
#   Muffin V1 -- VERSION 1 (code name: Fox)            #
#   Is not very good but it is okay??? IDK             #
########################################################

import random
import math
from collections import defaultdict

class TextGenerator:
    def __init__(self, corpus):
        self.chain = defaultdict(lambda: defaultdict(int))
        self.train(corpus)

    # Splits a string into words
    def split_words(self, input_text):
        return input_text.split()

    # Trains the model by building a transition map from the corpus (bigram approach)
    def train(self, corpus):
        words = self.split_words(corpus)
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            self.chain[current_word][next_word] += 1

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

    # Generates a sequence of words based on the Markov Chain model
    def generate(self, start_word, length, temperature, k):
        current_word = start_word
        result = [current_word]

        for _ in range(length):
            next_words = self.chain[current_word]
            if not next_words:
                break

            # Pick a random next word based on the bigram probabilities
            current_word = self.weighted_random(next_words, temperature, k)
            if not current_word:
                break
            result.append(current_word)

        # Continue generating until we get a sentence that ends with punctuation
        while not self.ends_with_punctuation(result[-1]):
            next_words = self.chain[result[-1]]
            if not next_words:
                break

            current_word = self.weighted_random(next_words, temperature, k)
            if not current_word:
                break
            result.append(current_word)

        return ' '.join(result)

    # Check if the last word ends with punctuation
    def ends_with_punctuation(self, word):
        return word[-1] in {'.', '!', '?'}


# Sample corpus for training (you can expand this with larger text)
corpus = """
The quick brown fox jumps over the lazy dog. 
The dog is lazy but the fox is quick. 
Foxes are fast and agile creatures. 
Dogs are loyal and friendly animals. 
In the forest, the fox hunts for food under the moonlight. 
The sly fox sneaks quietly, while the dog barks loudly at night. 
Many animals live in the forest, including foxes, birds, and deer. 
Foxes are known for their cleverness and ability to adapt to different environments. 
Dogs, on the other hand, have been human companions for centuries. 
They help humans by guarding homes, herding livestock, and providing companionship. 
Some dogs are trained for rescue missions, while others enjoy playing fetch in the park.
"""

# Initialize the text generator with the corpus
generator = TextGenerator(corpus)

# User input for generating text
user_input = "The"  # You can change this to any starting word
length = 100  # Length of the generated text
temperature = 0.835  # Adjust the randomness (1.0 is normal, lower is more deterministic) (Don't go below 0.835. please DO NOT FUCKING DARE TO GO BELOW 0.835)
k = 5  # Number of top choices to consider

# Generate text starting with the user input
generated_text = generator.generate(user_input, length, temperature, k)

print("Generated Text: " + generated_text)
