import json
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=2,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        return self.fc(output), hidden


def load_model():
    checkpoint = torch.load(
        "models/lstm_model.pth",
        map_location=DEVICE,
        weights_only=True
    )

    with open("models/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    char_to_idx = vocab["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}

    model = TextLSTM(
        checkpoint["vocab_size"],
        checkpoint["hidden_size"],
        checkpoint.get("embed_size", 128)
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, char_to_idx, idx_to_char


def generate_text(model, char_to_idx, idx_to_char, seed,
                  temperature=0.8, max_length=300):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if max_length < 1:
        raise ValueError("Maximum length must be at least 1.")

    seed = seed or "The"
    unknown = [ch for ch in seed if ch not in char_to_idx]
    if unknown:
        raise ValueError(
            "The starting text contains characters not found in the training corpus. "
            "Try a shorter seed using characters present in corpus.txt."
        )

    generated = seed
    input_ids = torch.tensor(
        [[char_to_idx[ch] for ch in seed]],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():
        logits, hidden = model(input_ids)

        for _ in range(max_length):
            next_logits = logits[:, -1, :] / temperature
            probabilities = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            generated += idx_to_char[next_id.item()]
            logits, hidden = model(next_id, hidden)

    return generated


if __name__ == "__main__":
    model, char_to_idx, idx_to_char = load_model()
    print(generate_text(
        model, char_to_idx, idx_to_char,
        seed="The stars", temperature=0.8, max_length=300
    ))
