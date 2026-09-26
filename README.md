## \--> Storyteller-Shakespeare

A GPT-style **decoder-only Transformer language model implemented from scratch using PyTorch** and trained on a Shakespeare text corpus for autoregressive text generation.

The project demonstrates the core components behind GPT-style language models — token embeddings, positional embeddings, causal self-attention, multi-head attention, feed-forward networks, residual connections, layer normalization, next-token prediction, and autoregressive generation — without relying on a pretrained language model.

!\[Project Screenshot](Screenshot.png)

\---

## \--> Project Overview

**Storyteller-Shakespeare** is a small-scale language model built to understand and demonstrate how a GPT-style Transformer can be implemented and trained from the ground up.

The model uses **character-level tokenization** and learns to predict the next character from a sequence of previously seen characters.

After training, the model can take a text prompt such as:

```text
ROMEO:
```

and generate a continuation based on patterns learned from the training corpus.

### What this project demonstrates

* Building a GPT-style Transformer from scratch
* Character-level tokenization
* Causal / masked self-attention
* Multi-head attention
* Positional embeddings
* Transformer residual connections
* Layer normalization
* Feed-forward neural networks
* Cross-entropy language-model training
* AdamW optimization
* GPU acceleration with CUDA
* Autoregressive text generation
* Train/validation loss evaluation
* Saving and loading PyTorch model checkpoints

\---

## \--> Model Architecture

The model follows a decoder-only Transformer architecture:

```text
                    Input Text
                        │
                        ▼
               Character Tokenization
                        │
                        ▼
                 Token Embeddings
                        +
               Positional Embeddings
                        │
                        ▼
             ┌──────────────────────┐
             │   Transformer Block  │
             │                      │
             │ LayerNorm            │
             │      ↓               │
             │ Multi-Head           │
             │ Self-Attention       │
             │      ↓               │
             │ Residual Connection  │
             │      ↓               │
             │ LayerNorm            │
             │      ↓               │
             │ Feed-Forward Network│
             │      ↓               │
             │ Residual Connection  │
             └──────────────────────┘
                        │
                 × 4 Transformer Blocks
                        │
                        ▼
                   Final LayerNorm
                        │
                        ▼
                  Language Model Head
                        │
                        ▼
                 Next-Character Logits
                        │
                        ▼
              Autoregressive Generation
                        │
                        ▼
                  Generated Text
```

\---

## \--> Model Configuration

|Parameter|Value|
|-|-:|
|Architecture|Decoder-only Transformer|
|Tokenization|Character-level|
|Vocabulary size|65 characters|
|Embedding dimension|128|
|Transformer layers|4|
|Attention heads|4|
|Context length|64 tokens|
|Dropout|0.2|
|Total parameters|816,705|
|Trainable parameters|816,705|
|Optimizer|AdamW|
|Learning rate|0.0003|
|Batch size|32|
|Training iterations|5,000|
|Loss function|Cross-Entropy Loss|
|Training hardware|NVIDIA GeForce RTX 3050 Laptop GPU|
|GPU acceleration|CUDA|

\---

## \--> Dataset

The model was trained on a Shakespeare text corpus stored in:

```text
data/input.txt
```

### Dataset statistics

* **Characters:** 1,115,394
* **Size:** approximately 1.06 MB
* **Vocabulary:** 65 unique characters

The dataset is divided into:

```text
90% → Training data
10% → Validation data
```

Character-level tokenization converts each character into an integer token before it is passed to the Transformer.

\---

## \--> Training Pipeline

The training process follows these steps:

```text
Raw Shakespeare Text
        ↓
Character Vocabulary
        ↓
Character → Integer Encoding
        ↓
Train / Validation Split
        ↓
Random Context Windows
        ↓
GPT Transformer
        ↓
Next-Token Prediction
        ↓
Cross-Entropy Loss
        ↓
Backpropagation
        ↓
AdamW Optimizer
        ↓
Updated Model Parameters
```

The training script evaluates training and validation loss periodically during training.

\---

## \--> Training Results

A fresh 5,000-iteration training run was performed using the NVIDIA RTX 3050 Laptop GPU.

|Metric|Result|
|-|-:|
|Training iterations|5,000|
|Training time|\~4 min 35 sec|
|Training loss|**1.6156**|
|Validation loss|**1.7824**|
|Device|CUDA|
|GPU|NVIDIA GeForce RTX 3050 Laptop GPU|

The reported evaluation loss was calculated using random batches from the training and validation splits.

### Baseline checkpoint

The previously trained checkpoint was also evaluated:

|Metric|Previous checkpoint|
|-|-:|
|Training loss|1.6092|
|Validation loss|1.7700|

The small difference between the two runs is expected because the evaluation samples are randomly selected and the model is relatively small.

\---

## \--> Text Generation

After training, the model can generate text from a user-provided prompt.

Run:

```bash
python generate.py
```

Example:

```text
Enter Story Prompt:
ROMEO:
```

The model then generates new text autoregressively, predicting one character at a time.

### Example generation

```text
ROMEO:
Peas is a would the lifer spetcet, be kind them silk.

SOMINIUS:

LEONTES:
Good strumbled, and good he request of prince and her woen
Rumbends thee!
```

The current model captures several Shakespeare-like patterns, including dialogue formatting, character names, punctuation, and vocabulary. Because this is a relatively small character-level model trained for 5,000 iterations, generated text can still contain malformed words and grammatical errors.

\---

## \--> Evaluation

The project includes an evaluation script:

```bash
python evaluate.py
```

It loads the saved model checkpoint and calculates average cross-entropy loss on randomly sampled batches from:

* Training data
* Validation data

This provides a quantitative measurement of model performance in addition to qualitative text-generation examples.

\---

## \--> Project Structure

```text
Storyteller-Shakespeare/
│
├── data/
│   └── input.txt              # Training corpus
│
├── model/
│   └── gpt.py                 # Transformer architecture
│
├── train.py                   # Model training pipeline
├── generate.py                # Text generation
├── evaluate.py                # Model evaluation
├── model.pth                  # Trained model checkpoint
│
│
├── Screenshot.png             # Project screenshot
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

\---

## \--> Getting Started

### 1\. Clone the repository

```bash
git clone https://github.com/Suman-Aryann/Storyteller-Shakespeare-.git
cd Storyteller-Shakespeare-
```

### 2\. Create a virtual environment

#### Windows

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

> For CUDA-enabled training, install a CUDA-compatible PyTorch build appropriate for your NVIDIA GPU and system. The repository's training code automatically selects CUDA when `torch.cuda.is\_available()` returns `True`.

\---

## \--> Generate Text Using the Existing Model

The repository includes a trained checkpoint.

Run:

```bash
python generate.py
```

Enter a prompt when requested.

Example:

```text
Enter Story Prompt:
Once upon a time
```

\---

## \--> Train the Model

To train the model from scratch:

```bash
python train.py
```

The script automatically selects:

```python
cuda
```

when CUDA is available; otherwise it falls back to CPU.

The training configuration is defined in `train.py` and the model architecture is defined in:

```text
model/gpt.py
```

After training, the resulting model checkpoint is saved as:

```text
model.pth
```

\---

## \--> Hardware

The model was trained using:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
CUDA: Enabled
```

The complete 5,000-iteration training run took approximately:

```text
4 minutes 35 seconds
```

on the development system.

\---

## \--> Tech Stack

* **Python**
* **PyTorch**
* **CUDA**
* **NumPy**
* **Matplotlib**
* **tqdm**

\---

## \--> Learning Objectives

This project was developed to gain practical understanding of:

1. Transformer architecture
2. GPT-style language modeling
3. Self-attention mechanisms
4. Causal masking
5. Token and positional embeddings
6. Neural language-model training
7. GPU-accelerated deep learning
8. Autoregressive text generation
9. Model evaluation
10. PyTorch model serialization

\---

## \--> Future Improvements

Potential improvements include:

* Larger model architecture
* Longer training
* Improved tokenization such as BPE
* Larger and more diverse training data
* Learning-rate scheduling
* Temperature and top-k/top-p sampling controls
* Web-based text-generation interface
* Interactive storytelling interface
* Training-loss visualization
* Hugging Face deployment

\---

## \--> Current Limitations

This project intentionally uses a relatively small Transformer architecture and character-level tokenization.

As a result:

* Generated text can contain malformed words
* Grammar and long-range coherence are limited
* The model has a small context window
* Generation quality is significantly below modern pretrained LLMs
* The model is intended as an educational / experimental implementation rather than a production language model

These limitations are useful for understanding the relationship between model size, training duration, dataset size, and language-generation quality.

\---

## \--> Author

**Suman Aryan**

GitHub:  
https://github.com/Suman-Aryann

\---

## 📌 Project Summary

**Storyteller-Shakespeare** is a from-scratch implementation of a GPT-style decoder-only Transformer in PyTorch.

The project contains the complete pipeline:

```text
Dataset
   ↓
Tokenization
   ↓
Transformer Architecture
   ↓
CUDA Training
   ↓
Model Checkpoint
   ↓
Evaluation
   ↓
Autoregressive Text Generation
```

The goal of the project is to demonstrate the underlying mechanics of GPT-style language models through an independently implemented and trained neural network rather than relying on a pretrained LLM API.

