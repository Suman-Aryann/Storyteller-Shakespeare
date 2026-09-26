import torch
from model.gpt import GPTLanguageModel

# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# =========================
# LOAD DATA
# =========================

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)

# =========================
# TRAIN / VALIDATION SPLIT
# =========================

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# =========================
# MODEL
# =========================

model = GPTLanguageModel(vocab_size)

model.load_state_dict(
    torch.load("model.pth", map_location=device)
)

model = model.to(device)
model.eval()

# =========================
# LOSS EVALUATION
# =========================

@torch.no_grad()
def evaluate(data, iterations=200):

    losses = []

    for _ in range(iterations):

        ix = torch.randint(
            len(data) - 64,
            (32,)
        )

        x = torch.stack([
            data[i:i+64] for i in ix
        ])

        y = torch.stack([
            data[i+1:i+65] for i in ix
        ])

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        losses.append(loss.item())

    return sum(losses) / len(losses)


train_loss = evaluate(train_data)
val_loss = evaluate(val_data)

print("\n========== MODEL EVALUATION ==========")
print(f"Training Loss   : {train_loss:.4f}")
print(f"Validation Loss : {val_loss:.4f}")
print("======================================")