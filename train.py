import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model.gpt import GPTLanguageModel


# ==========================================
# DEVICE
# ==========================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using Device:", device)


# ==========================================
# LOAD DATASET
# ==========================================

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# ==========================================
# TOKENIZATION
# ==========================================

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}

itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]

decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

print("Vocabulary Size:", vocab_size)


# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


# ==========================================
# HYPERPARAMETERS
# ==========================================

batch_size = 32
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4


# ==========================================
# BATCH FUNCTION
# ==========================================

def get_batch(split):

    data = train_data if split == 'train' else val_data

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    x, y = x.to(device), y.to(device)

    return x, y


# ==========================================
# LOSS ESTIMATION
# ==========================================

@torch.no_grad()
def estimate_loss():

    out = {}

    model.eval()

    for split in ['train', 'val']:

        losses = torch.zeros(200)

        for k in range(200):

            X, Y = get_batch(split)

            logits, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out


# ==========================================
# CREATE MODEL
# ==========================================

model = GPTLanguageModel(vocab_size)

model = model.to(device)

print("\nModel Created Successfully")


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ==========================================
# LOSS HISTORY
# ==========================================

train_losses = []
val_losses = []
steps = []


# ==========================================
# TRAINING LOOP
# ==========================================

print("\nStarting Training...\n")

for iter in range(max_iters):

    # --------------------------------------
    # Evaluate loss
    # --------------------------------------

    if iter % eval_interval == 0:

        losses = estimate_loss()

        train_loss = losses['train']
        val_loss = losses['val']

        steps.append(iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"step {iter}: "
            f"train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}"
        )

    # --------------------------------------
    # Get training batch
    # --------------------------------------

    xb, yb = get_batch('train')

    # --------------------------------------
    # Forward pass
    # --------------------------------------

    logits, loss = model(xb, yb)

    # --------------------------------------
    # Backpropagation
    # --------------------------------------

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()


# ==========================================
# FINAL LOSS EVALUATION
# ==========================================

final_losses = estimate_loss()

final_train_loss = final_losses['train']
final_val_loss = final_losses['val']

steps.append(max_iters)
train_losses.append(final_train_loss)
val_losses.append(final_val_loss)

print(
    f"step {max_iters}: "
    f"train loss {final_train_loss:.4f}, "
    f"val loss {final_val_loss:.4f}"
)

print("\nTraining Completed!")


# ==========================================
# TRAINING LOSS GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    steps,
    train_losses,
    marker='o',
    label='Training Loss'
)

plt.plot(
    steps,
    val_losses,
    marker='o',
    label='Validation Loss'
)

plt.xlabel("Training Iterations")
plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nTraining loss graph saved as training_loss.png")


# ==========================================
# TEXT GENERATION
# ==========================================

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

generated_tokens = model.generate(
    context,
    max_new_tokens=500
)[0].tolist()

generated_text = decode(generated_tokens)

print(
    "\n================ GENERATED TEXT ================\n"
)

print(generated_text)

print("\n================================================")


# ==========================================
# SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "model.pth"
)

print("\nModel Saved Successfully!")