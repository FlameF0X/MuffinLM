################################################################
#   Muffin V2.7l -- VERSION 2.7 large (code name: Elizabeth)   #
#   Now more BIG                                               #
################################################################

import os
import random
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CorpusDataset(Dataset):
    def __init__(self, data: List[str], seq_length: int):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        input_seq = self.data[index:index + self.seq_length]
        target_seq = self.data[index + 1:index + self.seq_length + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)


class TextGeneratorNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int):
        super(TextGeneratorNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden


class TextGenerator:
    def __init__(self, corpus_path: str, seq_length: int = 20, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2) -> None:
        self.seq_length = seq_length
        self.corpus = self.load_corpus(corpus_path)
        self.words = self.split_words(self.corpus)
        self.vocab = list(set(self.words))  # Unique words
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        self.model = TextGeneratorNN(len(self.vocab), embedding_dim, hidden_dim, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

        # Prepare dataset and dataloader
        corpus_indices = [self.word_to_idx[word] for word in self.words]
        self.dataset = CorpusDataset(corpus_indices, self.seq_length)
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)

        # Directory for saving/loading model
        self.model_path = 'model/model-main.pth'
        self.training_dir = 'model'

        # Ensure the directory exists
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)

        # Check if the model file exists
        if os.path.exists(self.model_path):
            print("Loading saved model from:", self.model_path)
            self.load_model()
        else:
            print("No saved model found. Training from scratch.")

    def load_corpus(self, file_path: str) -> str:
        """Load the corpus from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_words(self, input_text: str) -> List[str]:
        """Split a string into words."""
        return input_text.split()

    def train(self, epochs: int = 10) -> None:
        """Train the neural network."""
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for input_seq, target_seq in self.dataloader:
                input_seq, target_seq = input_seq.long(), target_seq.long()
                self.optimizer.zero_grad()

                output, _ = self.model(input_seq)
                loss = self.loss_fn(output.view(-1, len(self.vocab)), target_seq.view(-1))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(self.dataloader)}")

        # Save the model after training
        print("Saving trained model to:", self.model_path)
        self.save_model()

    def generate(self, start_words: str, length: int, temperature: float) -> str:
        self.model.eval()

        current_words = start_words.split()
        input_seq = torch.tensor([self.word_to_idx[word] for word in current_words]).unsqueeze(0)

        hidden = None
        result = current_words[:]

        for _ in range(length):
            with torch.no_grad():
                output, hidden = self.model(input_seq, hidden)

            probabilities = torch.softmax(output[:, -1, :] / temperature, dim=-1).squeeze()
            next_word_idx = torch.multinomial(probabilities, 1).item()
            next_word = self.idx_to_word[next_word_idx]

            result.append(next_word)
            input_seq = torch.tensor([next_word_idx]).unsqueeze(0)

        # Continue generating until we hit punctuation after reaching the length limit
        while not self.ends_with_punctuation(result[-1]):
            with torch.no_grad():
                output, hidden = self.model(input_seq, hidden)

            probabilities = torch.softmax(output[:, -1, :] / temperature, dim=-1).squeeze()
            next_word_idx = torch.multinomial(probabilities, 1).item()
            next_word = self.idx_to_word[next_word_idx]

            result.append(next_word)
            input_seq = torch.tensor([next_word_idx]).unsqueeze(0)

        return ' '.join(result)

    @staticmethod
    def ends_with_punctuation(word: str) -> bool:
        """Check if the word ends with punctuation."""
        return word[-1] in {'.', '!', '?'}

    def get_random_starting_words(self, word_count: int = 2) -> str:
        """Select random starting words that exist in the corpus."""
        if len(self.words) < word_count:
            raise ValueError("Not enough words in the corpus for starting sequence.")
        start_index = random.randint(0, len(self.words) - word_count)
        return ' '.join(self.words[start_index:start_index + word_count])

    def save_model(self):
        """Save the trained model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
        }, self.model_path)

    def load_model(self):
        """Load the saved model and optimizer state."""
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))  # Add map_location
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.vocab = checkpoint['vocab']
        self.word_to_idx = checkpoint['word_to_idx']
        self.idx_to_word = checkpoint['idx_to_word']

    def save_generated_text(self, text: str, file_path: str = './SaveGeneratedText.txt') -> None:
        """Save the generated text to a specified file."""
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(text + '\n')  # Append the text followed by a newline


# Use the larger corpus dataset (dataset-4.txt)
corpus_file_path = 'dataset/dataset-5-large.txt'

# Initialize the text generator with the LSTM model
generator = TextGenerator(corpus_file_path)

# If model doesn't exist, train the neural network model (adjust epochs as needed)
if not os.path.exists(generator.model_path):
    generator.train(epochs=50)

# Loop to generate text until the user decides to save it
while True:
    # Randomly select starting words from the dataset
    start_words = generator.get_random_starting_words(word_count=3)
    length = 100  # Length of the generated text
    temperature = 0.835  # Adjust the randomness (0.835)

    # Generate text starting with the randomly selected start_words
    generated_text = generator.generate(start_words, length, temperature)

    print("Starting Words: " + start_words)
    print("Generated Text: " + generated_text)

    # Prompt to save the generated text
    save_choice = input(">> Do you want to save the generated text? (yes/no/cancel/stop): ").strip().lower()
    if save_choice == 'yes':
        generator.save_generated_text(generated_text)
        print("Generated text saved to './SaveGeneratedText.txt'.")

    elif save_choice == 'no':
        print("Generating a new text...")
    elif save_choice in ('cancel', 'stop'):
        print("Operation cancelled.")
        break
    else:
        print("Invalid input. Please respond with 'yes', 'no' or 'cancel'/'stop'.")
