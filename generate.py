import torch

from model.gpt import GPTLanguageModel

# ==========================================
# DEVICE
# ==========================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# LOAD DATASET
# ==========================================

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }

itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]

decode = lambda l: ''.join([itos[i] for i in l])

# ==========================================
# LOAD MODEL
# ==========================================

model = GPTLanguageModel(vocab_size)

model.load_state_dict(
    torch.load("model.pth", map_location=device)
)

model = model.to(device)

model.eval()

print("Model Loaded Successfully!")

# ==========================================
# USER PROMPT
# ==========================================

prompt = input("\nEnter Story Prompt:\n")

context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device
)

# ==========================================
# GENERATE TEXT
# ==========================================

generated_tokens = model.generate(
    context,
    max_new_tokens=300
)[0].tolist()

generated_text = decode(generated_tokens)

print("\n================ STORY ================\n")

print(generated_text)

print("\n=======================================")