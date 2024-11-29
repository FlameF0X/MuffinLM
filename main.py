################################################################
#   Muffin V2.7l -- VERSION 2.7 large                          #
#                                                              #
#    We all love Muffin 2.7!                                   #
#                                                              #
#    architecture: LSTM <3                                     #
################################################################

import os
import random
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CorpusDataset(Dataset):
    def __init__(self, data: List[int], seq_length: int):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index: int):
        input_seq = self.data[index:index + self.seq_length]
        target_seq = self.data[index + 1:index + self.seq_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


class TextGeneratorNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
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
        self.corpus = self._load_corpus(corpus_path)
        self.words = self.corpus.split()
        self.vocab = list(set(self.words))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        corpus_indices = [self.word_to_idx[word] for word in self.words]
        self.dataset = CorpusDataset(corpus_indices, self.seq_length)
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=2)

        self.model = TextGeneratorNN(len(self.vocab), embedding_dim, hidden_dim, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

        self.model_path = 'model-main.pth'
        os.makedirs('model', exist_ok=True)

        if os.path.exists(self.model_path):
            print("Loading saved model...")
            self._load_model()
        else:
            print("No saved model found. Training from scratch.")

    def _load_corpus(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def train(self, epochs: int = 10) -> None:
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for input_seq, target_seq in self.dataloader:
                input_seq, target_seq = input_seq.to(torch.long), target_seq.to(torch.long)
                self.optimizer.zero_grad()

                output, _ = self.model(input_seq)
                loss = self.loss_fn(output.view(-1, len(self.vocab)), target_seq.view(-1))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        print("Saving trained model...")
        self._save_model()

    def generate(self, start_words: str, length: int, temperature: float) -> str:
        self.model.eval()
        current_words = start_words.split()
        input_seq = torch.tensor([self.word_to_idx[word] for word in current_words], dtype=torch.long).unsqueeze(0)
        hidden = None
        result = current_words[:]

        with torch.no_grad():
            for _ in range(length):
                output, hidden = self.model(input_seq, hidden)
                probabilities = torch.softmax(output[:, -1, :] / temperature, dim=-1).squeeze()
                next_word_idx = torch.multinomial(probabilities, 1).item()
                next_word = self.idx_to_word[next_word_idx]
                result.append(next_word)
                input_seq = torch.tensor([[next_word_idx]], dtype=torch.long)

        return ' '.join(result)

    def save_generated_text(self, text: str, file_path: str = './SaveGeneratedText.txt') -> None:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(text + '\n')

    def _save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
        }, self.model_path)

    def _load_model(self):
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.vocab = checkpoint['vocab']
        self.word_to_idx = checkpoint['word_to_idx']
        self.idx_to_word = checkpoint['idx_to_word']


if __name__ == "__main__":
    corpus_file_path = 'dataset-5-large.txt'
    generator = TextGenerator(corpus_file_path)

    if not os.path.exists(generator.model_path):
        generator.train(epochs=50)

    while True:
        start_words = generator.words[random.randint(0, len(generator.words) - 3):][:3]
        length = 100
        temperature = 0.835
        generated_text = generator.generate(" ".join(start_words), length, temperature)

        print("\nGenerated Text:")
        print(generated_text)

        save_choice = input("Save this text? (yes/no/stop): ").strip().lower()
        if save_choice == 'yes':
            generator.save_generated_text(generated_text)
            print("Text saved.")
        elif save_choice == 'stop':
            print("Stopping generation.")
            break
