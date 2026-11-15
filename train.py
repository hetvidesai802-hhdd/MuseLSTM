import os
import json
import numpy as np
import torch
import torch.nn as nn

DATA_PATH = "dataset/corpus.txt"
MODEL_PATH = "models/lstm_model.pth"
VOCAB_PATH = "models/vocab.json"

SEQ_LENGTH = 80
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.002
HIDDEN_SIZE = 256
EMBED_SIZE = 128
STEPS_PER_EPOCH = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size=EMBED_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        return self.fc(output), hidden


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    if len(text) <= SEQ_LENGTH + 1:
        raise ValueError(
            f"Corpus is too short. Add more text (at least {SEQ_LENGTH + 2} characters)."
        )

    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    encoded = np.array([char_to_idx[ch] for ch in text], dtype=np.int64)
    vocab_size = len(chars)

    print(f"Device: {device}")
    print(f"Characters: {len(text)} | Vocabulary: {vocab_size}")

    model = TextLSTM(vocab_size, HIDDEN_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    max_start = len(encoded) - SEQ_LENGTH - 1
    steps = min(STEPS_PER_EPOCH, max(10, len(encoded) // BATCH_SIZE))

    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for _ in range(steps):
            starts = np.random.randint(0, max_start + 1, size=BATCH_SIZE)
            x = np.stack([encoded[i:i + SEQ_LENGTH] for i in starts])
            y = np.stack([encoded[i + 1:i + SEQ_LENGTH + 1] for i in starts])

            x = torch.tensor(x, dtype=torch.long, device=device)
            y = torch.tensor(y, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / steps
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] - Loss: {average_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "hidden_size": HIDDEN_SIZE,
        "embed_size": EMBED_SIZE,
        "seq_length": SEQ_LENGTH
    }, MODEL_PATH)

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "char_to_idx": char_to_idx,
            "idx_to_char": {str(k): v for k, v in idx_to_char.items()}
        }, f, ensure_ascii=False, indent=2)

    print("\nTraining completed successfully!")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Vocabulary saved: {VOCAB_PATH}")


if __name__ == "__main__":
    main()
