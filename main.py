################################################################
#   Muffin V2.8 -- VERSION 2.8                                 #
#                                                              #
#    Warming!!! This AI model is trined on r/AskReddit and     #
#    is not censored !!!                                       #
#                                                              #
#    architecture: LSTM <3                                     #
################################################################

import os
import random
import json
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class DialogDataset(Dataset):
    def __init__(self, dialogs: List[Dict[str, str]], word_to_idx: Dict[str, int], seq_length: int):
        self.dialogs = dialogs
        self.word_to_idx = word_to_idx
        self.seq_length = seq_length

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        input_text = dialog['pers_1']
        target_text = dialog['pers_2']

        input_seq = [self.word_to_idx[word] for word in input_text.split()]
        target_seq = [self.word_to_idx[word] for word in target_text.split()]

        # Ensure fixed sequence length
        input_seq = self.pad_sequence(input_seq, self.seq_length)
        target_seq = self.pad_sequence(target_seq, self.seq_length)

        return torch.tensor(input_seq), torch.tensor(target_seq)

    def pad_sequence(self, sequence: List[int], length: int, pad_token: int = 0) -> List[int]:
        """Pads sequences with `pad_token` to match the desired length."""
        if len(sequence) < length:
            return sequence + [pad_token] * (length - len(sequence))
        return sequence[:length]


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
    def __init__(self, json_path: str, seq_length: int = 20, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 4) -> None:
        self.seq_length = seq_length
        self.dialogs = self.load_json(json_path)
        self.words = self.extract_vocab(self.dialogs)
        self.vocab = list(set(self.words))  # Unique words
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        self.model = TextGeneratorNN(len(self.vocab), embedding_dim, hidden_dim, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

        # Prepare dataset and dataloader
        self.dataset = DialogDataset(self.dialogs, self.word_to_idx, self.seq_length)
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)

        # Directory for saving/loading model
        self.model_path = 'model-main.pth'

        # Check if the model file exists
        if os.path.exists(self.model_path):
            print("Loading saved model from:", self.model_path)
            self.load_model()
        else:
            print("No saved model found. Training from scratch.")

    def load_json(self, file_path: str) -> List[Dict[str, str]]:
        """Load dialog dataset from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def extract_vocab(self, dialogs: List[Dict[str, str]]) -> List[str]:
        """Extract vocabulary from dialogs."""
        vocab = []
        for dialog in dialogs:
            vocab.extend(dialog['pers_1'].split())
            vocab.extend(dialog['pers_2'].split())
        return vocab

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

    def generate(self, start_text: str) -> str:
        # Fixed values for response length and temperature
        length = 15  # Length of the generated response
        temperature = 0.835  # Temperature for randomness

        self.model.eval()

        input_seq = [self.word_to_idx.get(word, 0) for word in start_text.split()]
        input_seq = torch.tensor(input_seq).unsqueeze(0)

        hidden = None
        result = start_text.split()

        for _ in range(length):
            with torch.no_grad():
                output, hidden = self.model(input_seq, hidden)

            probabilities = torch.softmax(output[:, -1, :] / temperature, dim=-1).squeeze()
            next_word_idx = torch.multinomial(probabilities, 1).item()
            next_word = self.idx_to_word[next_word_idx]

            result.append(next_word)
            input_seq = torch.tensor([[next_word_idx]])

        return ' '.join(result)

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
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.vocab = checkpoint['vocab']
        self.word_to_idx = checkpoint['word_to_idx']
        self.idx_to_word = checkpoint['idx_to_word']


# Load dataset and initialize the generator
json_file_path = 'dialogue_dataset.json'

# Initialize the text generator with the JSON dataset
generator = TextGenerator(json_file_path)

# If model doesn't exist, train the neural network model
if not os.path.exists(generator.model_path):
    generator.train(epochs=300)

# Example usage of the generator
while True:
    start_text = input("Enter the starting text for pers_1: ")

    # Generate response with fixed values
    generated_text = generator.generate(start_text)
    print("Generated Text (pers_2):", generated_text)

    save_choice = input("Do you want to save this text? (yes/no): ").strip().lower()
    if save_choice == 'yes':
        with open('generated-dialogs.txt', 'a', encoding='utf-8') as file:
            file.write(f"pers_1: {start_text}\n")
            file.write(f"pers_2: {generated_text}\n\n")
        print("Saved!")
    elif save_choice == 'no':
        continue
    else:
        break
