# --> Storyteller-Shakespeare

A small GPT-style **character-level Transformer language model** implemented from scratch using PyTorch and trained on Shakespearean text.

The project demonstrates the core workflow of a decoder-only Transformer language model, including character-level tokenization, self-attention, Transformer blocks, autoregressive next-character prediction, GPU-accelerated training, evaluation, and text generation.

The model was trained using **CUDA on an NVIDIA RTX 3050 Laptop GPU**.

\---

## --> Project Overview

The goal of this project is to understand and implement the fundamental components behind GPT-style language models rather than relying on a pre-trained language model API.

* Character-level tokenization
* Token and positional embeddings
* Causal self-attention
* Multi-head attention
* Feed-forward neural networks
* Residual connections
* Layer normalization
* Decoder-only Transformer blocks
* Autoregressive text generation
* Cross-entropy language-model training
* AdamW optimization
* CUDA/GPU-accelerated training
* Training and validation loss evaluation

The model is intentionally compact so that it can be trained locally on a consumer laptop GPU.

## --> Model Architecture

The model uses a decoder-only Transformer architecture.

!\[Model Architecture](architecture\_diagram.png)

The model predicts the next character based on the previous characters in a fixed context window.

### --> Architecture Configuration

Parameter                Value

\---

Architecture             Decoder-only Transformer
Vocabulary Size          65 characters
Embedding Dimension      128
Transformer Layers       4
Attention Heads          4
Context Length           64
Dropout                  0.2
Feed-Forward Dimension   512
Parameters               816,705

## --> Dataset

The training corpus is stored at:

&#x20;   data/input.txt



* Size: approximately 1.06 MB
* Characters: approximately 1.1 million
* Vocabulary: 65 unique characters
* Training split: 90%
* Validation split: 10%

The model uses character-level tokenization, meaning each individual character is represented as a token.

## --> Training Pipeline

&#x20;   Shakespeare Text
|
v
Character Vocabulary
|
v
Character Encoding
|
v
Train / Validation Split
|
v
Random Context Batches
|
v
Token + Position Embeddings
|
v
4 Transformer Blocks
|
v
Causal Multi-Head Self-Attention
|
v
Feed-Forward Network
|
v
Linear Language Model Head
|
v
Next-Character Prediction
|
v
Cross-Entropy Loss
|
v
Backpropagation + AdamW
|
v
CUDA Training
|
v
model.pth



## --> Training Configuration

Parameter             Value

\---

Batch Size            32
Context Length        64
Training Iterations   5,000
Evaluation Interval   500 iterations
Learning Rate         0.0003
Optimizer             AdamW
Loss Function         Cross-Entropy
Training Device       NVIDIA RTX 3050 Laptop GPU
Framework             PyTorch
Accelerator           CUDA

## --> Training Results

The latest 5,000-iteration training run was completed on the NVIDIA RTX 3050 Laptop GPU.

Metric                Latest Result

\---

Training Loss         1.6096
Validation Loss       1.7914
Training Iterations   5,000

The training loss decreased substantially during training, while the validation loss also decreased and remained slightly higher than the training loss.

### --> Training and Validation Loss

!\[Training and Validation Loss](training\_loss.png)

The graph above was generated directly from the loss measurements recorded during the training run.

## --> Text Generation

After training, the saved model can generate Shakespeare-style text from a user-provided prompt.

Example prompts tested during development:

&#x20;   Once upon a time
The king entered the castle
ROMEO:
HAMLET:
In the dark forest



Example generation using the trained model:

&#x20;   ROMEO:

&#x20;   A clive the mastraw, the sair, the cortones that wat bray
That prince be more the courtes, my like for I contly breat,
And my like that is be for the pray of the than than of him.





The model learns Shakespeare-like vocabulary, dialogue formatting, and character names, but the current small character-level model can still produce malformed words, inconsistent grammar, and less coherent passages.

## --> Evaluation

The repository includes `evaluate.py`.

&#x20;   python evaluate.py



The latest checkpoint produced:

&#x20;   Training Loss    : 1.6096
Validation Loss  : 1.7914



Evaluation can run on CPU or CUDA depending on the available PyTorch environment.

## --> Project Structure

&#x20;   Storyteller-Shakespeare/
|
+-- data/
|   +-- input.txt
|
+-- model/
|   +-- gpt.py
|
+-- architecture\_diagram.png
+-- training\_loss.png
+-- .gitignore
+-- README.md
+-- Screenshot.png
+-- evaluate.py
+-- generate.py
+-- model.pth
+-- requirements.txt
+-- train.py



File                         Purpose

\---

`train.py`                   Trains the Transformer and records training/validation loss
`generate.py`                Generates text from the trained model
`evaluate.py`                Evaluates training and validation loss
`model/gpt.py`               Defines the GPT-style Transformer architecture
`data/input.txt`             Shakespeare training corpus
`model.pth`                  Trained model checkpoint
`architecture\\\\\\\_diagram.png`   Visual representation of the model architecture
`training\\\\\\\_loss.png`          Training and validation loss graph
`requirements.txt`           Python dependencies
`Screenshot.png`             Project/output screenshot
`.gitignore`                 Prevents unnecessary local files from being committed

## --> Getting Started

### --> 1. Clone the repository

&#x20;   git clone https://github.com/Suman-Aryann/Storyteller-Shakespeare.git
cd Storyteller-Shakespeare



### --> 2. Create a virtual environment

Windows:

&#x20;   python -m venv venv
venv\\Scripts\\activate



Linux/macOS:

&#x20;   python3 -m venv venv
source venv/bin/activate



### --> 3. Install dependencies

&#x20;   pip install -r requirements.txt



## --> Check CUDA

&#x20;   python -c "import torch; print(torch.cuda.is\_available()); print(torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else 'CPU')"



The model was trained successfully on an NVIDIA RTX 3050 Laptop GPU using CUDA.

## --> Generate Text

&#x20;   python generate.py



The script loads `model.pth` and prompts the user for a starting text.

&#x20;   Enter Story Prompt:
ROMEO:



## --> Train the Model

&#x20;   python train.py



* 5,000 training iterations
* Batch size 32
* Context length 64
* Learning rate 0.0003
* AdamW optimizer
* CUDA when available
* Loss evaluation every 500 iterations

The training script also generates `training\\\\\\\_loss.png` after training.

## --> Evaluate the Model

&#x20;   python evaluate.py



This calculates the model's training and validation loss using the saved checkpoint.

## --> Hardware Used

Component     Specification

\---

CPU           AMD Ryzen 5 5600H
GPU           NVIDIA RTX 3050 Laptop GPU
RAM           16 GB
Framework     PyTorch
Accelerator   CUDA

## --> Technologies Used

* Python
* PyTorch
* CUDA
* Transformer Architecture
* Multi-Head Self-Attention
* Deep Learning
* Character-Level Language Modeling
* Matplotlib
* Git
* GitHub

## --> What I Learned

1. Character-level tokenization
2. Vocabulary creation
3. Training/validation dataset splitting
4. Context-window sampling
5. Token and positional embeddings
6. Causal self-attention
7. Multi-head attention
8. Feed-forward neural networks
9. Residual connections
10. Layer normalization
11. Transformer blocks
12. Autoregressive language modeling
13. Cross-entropy loss
14. AdamW optimization
15. GPU-accelerated training
16. Model checkpointing
17. Text generation
18. Training/validation loss evaluation

## --> Limitations

* Small model size
* Character-level tokenization
* Short context window
* Limited training iterations
* Limited training corpus
* Occasional malformed words
* Inconsistent grammar
* Limited long-range context
* No instruction tuning
* No modern subword tokenizer
* No large-scale pretraining

Because of these constraints, generated text should be viewed as an experimental demonstration of Transformer-based language modeling.

## --> Future Improvements

### --> Model Improvements

* Increase embedding dimension
* Increase the number of Transformer layers
* Increase the number of attention heads
* Increase context length
* Train for more iterations
* Tune learning rate and dropout
* Add learning-rate scheduling

### --> Tokenization Improvements

* Implement BPE tokenization
* Experiment with subword tokenization
* Compare character-level and token-level generation

### --> Dataset Improvements

* Use a larger and cleaner corpus
* Increase dataset diversity
* Experiment with additional literary datasets

### --> Generation Improvements

* Add temperature control
* Add top-k sampling
* Add top-p / nucleus sampling
* Add configurable maximum generation length
* Add an interactive web interface

### --> Engineering Improvements

* Add experiment tracking
* Add automated evaluation
* Add configuration files
* Add Docker support
* Deploy an inference API

## --> Why a Small GPT Model?

Large language models require substantial computational resources for training. This project focuses instead on understanding the fundamental components that make GPT-style models work.

* Inspect the architecture
* Modify the implementation
* Train locally
* Experiment with hyperparameters
* Observe training behavior
* Understand autoregressive generation

## --> Reproducibility

The repository contains the trained checkpoint `model.pth`, allowing the generation script to run without retraining.

&#x20;   python train.py
python evaluate.py
python generate.py



The repository was tested from a fresh clone with a new virtual environment. The dependencies installed successfully, the checkpoint loaded, a prompt was accepted, text was generated, and the program exited normally.

## --> Project Screenshot

!\[Project Screenshot](Screenshot.png)

## --> Learning Objective

The primary objective of this project was to move beyond using pre-built language-model APIs and understand the underlying mechanics of a GPT-style architecture through implementation.

## --> Author

**Suman Aryan**

GitHub: https://github.com/Suman-Aryann

## --> Project Summary

**Storyteller-Shakespeare** is a compact GPT-style character-level language model implemented in PyTorch and trained on Shakespearean text.

The project demonstrates how a decoder-only Transformer can learn character-level language patterns and generate new Shakespeare-style text from a prompt.

The model contains **816,705 parameters** and was trained for **5,000 iterations on an NVIDIA RTX 3050 Laptop GPU using CUDA**.

The project is primarily focused on learning, experimentation, and demonstrating the fundamentals of Transformer-based language modeling.
