import torch
import torch.nn as nn

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

stoi = { ch:i for i,ch in enumerate(chars) }

itos = { i:ch for i,ch in enumerate(chars) }

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
        data[i:i+block_size] for i in ix
    ])

    y = torch.stack([
        data[i+1:i+block_size+1] for i in ix
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

        out[split] = losses.mean()

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
# TRAINING LOOP
# ==========================================

print("\nStarting Training...\n")

for iter in range(max_iters):

    # Evaluate loss
    if iter % eval_interval == 0:

        losses = estimate_loss()

        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    # Get batch
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

print("\nTraining Completed!")

# ==========================================
# TEXT GENERATION
# ==========================================

context = torch.zeros((1,1), dtype=torch.long, device=device)

generated_tokens = model.generate(
    context,
    max_new_tokens=500
)[0].tolist()

generated_text = decode(generated_tokens)

print("\n================ GENERATED TEXT ================\n")

print(generated_text)

print("\n================================================")


torch.save(model.state_dict(), "model.pth")

print("\nModel Saved Successfully!")