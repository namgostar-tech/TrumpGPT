# TrumpGPT: When Rhetoric Outpaces Reasoning in Language Models

> **A decoder-only language model trained from scratch to explore the boundary where stylistic mimicry decouples from semantic reasoning.**

[![Article](https://img.shields.io/badge/Read_The_Full_Article-namgostar.com-red?style=flat-square)](https://namgostar.com/projects/trumpgpt)
[![Follow-up Project](https://img.shields.io/badge/Phase_2-Trump--Llama--3.1--8B--Instruct-blue?style=flat-square)](https://github.com/namgostar-tech/Trump-Llama-3.1-8B-Instruct)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview & Research Thesis

If you are watching nations scramble for superiority in an AI arms race while authoritarianism spreads globally, you might be morbidly curious why someone would train a custom language model on Donald Trump’s Truth Social profile.

Authoritarian regimes place specific emphasis on perfecting their rhetoric, and autoregressive language models excel at automating this process. **TrumpGPT** was developed as an experimental AI trust & safety artifact to explore what certain political figures and language models share: **the ability to execute style mimicry and persuasion, while falling short of logical comprehension and semantic reasoning.**

This project demonstrates how small decoder-only transformers readily master the rhetorical "vibe"—capitalization habits, punctuation rhythm, and polarized vocabulary clusters—while completely failing to maintain semantic logic across extended generation lengths.

---

## The Research Series

This project is the first entry in a two-part research series on rhetorical emulation in language models:

| Project | Paradigm | Architecture | Primary Capability | Key Research Finding |
| :--- | :--- | :--- | :--- | :--- |
| **[TrumpGPT](https://github.com/namgostar-tech/TrumpGPT)** *(Phase 1)* | Trained from Scratch | 12-layer Decoder Transformer (`nanoGPT` base + FlashAttention) | Unconditional Text Generation | Style and semantics decouple: model learns stylistic markers (20% ALL CAPS, polarized PMI associations) but collapses into word salad over long contexts. |
| **[Trump-Llama-3.1-8B-Instruct](https://github.com/namgostar-tech/Trump-Llama-3.1-8B-Instruct)** *(Phase 2)* | Instruction Fine-Tuned (QLoRA) | 8B Foundation Model (`Llama-3.1-8B-Instruct`) | Conversational & Interactive Prompting | Foundation models maintain conversational thread and topic coherence via pre-trained latent space, yet deeper scrutiny reveals rhetoric still supersedes true reasoning. |

📖 **Read the full essay and interactive breakdown:** [TrumpGPT: When Rhetoric Outpaces Reasoning in LMs](https://namgostar.com/projects/trumpgpt) on Roshan Namgostar's portfolio.

---

## Model Architecture

TrumpGPT scales up Andrej Karpathy's foundational [`nanoGPT`](https://github.com/karpathy/nanoGPT) decoder-only architecture with modern performance optimizations:

```
[ Input Tokens ] ---> [ Token & Position Embeddings (d=768) ]
                              │
                    ┌─────────▼─────────┐
                    │  LayerNorm 1      │
                    │  Causal FlashAttn │ (12 Heads, Pre-Norm)
                    │  Residual (+)     │
                    ├───────────────────┤  x 12 Transformer Blocks
                    │  LayerNorm 2      │
                    │  MLP (4x hidden)  │ (Dropout = 0.2)
                    │  Residual (+)     │
                    └─────────┬─────────┘
                              │
                    [ LayerNorm (Final) ]
                              │
                    [ LM Head (Linear)  ] ---> [ Next-Token Logits ]
```

### Technical Specifications
- **Architecture**: Decoder-only autoregressive transformer (GPT-2 style).
- **Tokenization**: Subword BPE via OpenAI's `tiktoken` (`gpt2` encoding, 50,257 vocabulary size) with `<|endoftext|>` delimiters.
- **Model Dimensions**: 
  - Embedding dimension (`n_embd`): **768**
  - Attention heads (`n_head`): **12** (head dimension: 64)
  - Transformer layers (`n_layer`): **12**
  - Context window (`block_size`): **256 tokens**
- **Attention Optimization**: PyTorch's native `F.scaled_dot_product_attention(..., is_causal=True)` utilizing fused memory-efficient FlashAttention kernels.
- **Regularization**: Dropout set to `0.2` and pre-norm `LayerNorm` formulation for training stability.

---

## Dataset & Training Pipeline

### 1. Data Cleaning (`extract_clean_truths.py`)
The dataset was extracted from a snapshot of Matt Stiles' [Trump Truth Social Archive](https://github.com/stiles/trump-truth-social-archive) containing 33,755 posts spanning February 2022 to June 2026. The extraction script filters noise to isolate pure author voice:
- Removes empty posts, link-only contributions, and retweets (`RT @...`).
- Decodes HTML entities and normalizes whitespace into single-line strings.
- Appends `<|endoftext|>` token delimiters.
- Yields **11,583 cleaned training posts** (`clean_truths.txt`, ~4.4 MB).

### 2. Training Dynamics (`train.py`)
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, $\text{lr}=3\times 10^{-4}$).
- **Precision**: Autocast mixed precision (`bfloat16`/`float16`) with `high` matrix multiplication precision.
- **Learning Rate Schedule**: Linear warmup (200 steps) followed by Cosine Annealing decay down to $3\times 10^{-5}$.
- **Gradient Clipping**: Norm clipped at `1.0`.
- **Early Stopping**: Monitored validation loss (90/10 train-val split) with a patience threshold of 5 evaluation intervals (evaluated every 250 steps).
- **Outcome**: The model reached optimal validation loss of **1.412** at step 1,500, producing a ~650 MB checkpoint (`TrumpGPT-Base-v1.pth`).

---

## Quantitative Probing & Findings

Using `probe_model.py`, 1,000 independent samples were analyzed against the training ground truth:

### 1. Stylistic Transfer Metrics
The model accurately absorbed—and even exaggerated—the stylistic markers of its training data:

| Metric | Training Dataset (`clean_truths.txt`) | TrumpGPT Generations (`1000_generations.txt`) | Analysis |
| :--- | :---: | :---: | :--- |
| **ALL CAPS Ratio** | **10.15%** | **20.01%** | Model over-indexes on uppercase words as an authoritative stylistic tool. |
| **Exclamation Frequency** | **24.65** / 1k words | **25.48** / 1k words | Nearly identical frequency of high-energy exclamation punctuation. |

### 2. Word Association Probes (Pointwise Mutual Information)
Pointwise Mutual Information (PMI) probes revealed that while the original dataset associates political groups with specific narrative facts, TrumpGPT reduced associations to pure sensationalist epithets:

- **Enemies (Democrats, Biden, Judges, DOJ)**: Strongly clustered around insults and labels (*"crooked"*, *"radical"*, *"comrade"*, *"joe"*, *"liz"*, *"merchan"*).
- **Allies (Republicans, MAGA, Patriots)**: Clustered around conflict and victory terms (*"landslide"*, *"indictments"*, *"viciously"*, *"matter"*).

### 3. Semantic Breakdown ("The Stochastic Parrot")
While short phrases sound unmistakably authentic, longer sequences quickly devolve into structural word salads lacking logical threads:

> *"I believe Democrats and Television. Few go for my campaign. They stole the DOJ and Crooked Democrats cheat to roam me in Ottawa of consecutive times in a landslide, right now, and he will. He should be immediately advance by this TRUMP DED. A loser SCAM from the ELECTION INTERFERENCE!..."*

---

## Repository Structure

```
TrumpGPT/
├── extract_clean_truths.py  # Cleans raw JSON archive into clean_truths.txt
├── train.py                 # Core model architecture, training loop, and generation
├── generate.py              # Standalone CLI text generation script
├── probe_model.py           # PMI analysis and stylistic metric verification tool
├── truth_archive.json       # Raw Truth Social post archive
├── clean_truths.txt         # Preprocessed single-line training dataset
├── 1000_generations.txt     # 1,000 sample generations used for research analysis
├── analysis_report.md       # Quantitative summary of probing findings
├── requirements.txt         # Python environment dependencies
└── README.md                # Project documentation
```

---

## Quickstart & Usage

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/namgostar-tech/TrumpGPT.git
cd TrumpGPT

# Install dependencies
pip install -r requirements.txt
```

### 2. Extract Dataset (Optional)
If you wish to re-process `truth_archive.json`:
```bash
python extract_clean_truths.py --input truth_archive.json --output clean_truths.txt
```

### 3. Train the Model
```bash
# Train on CUDA, Apple Silicon (MPS), or CPU
python train.py --mode train --data_file clean_truths.txt --weights_path TrumpGPT-Base-v1.pth
```

### 4. Generate Text
Generate unprompted or prompted text from trained weights:
```bash
# Unconditional generation
python generate.py --weights_path TrumpGPT-Base-v1.pth --max_new_tokens 300

# Prompted generation
python generate.py --weights_path TrumpGPT-Base-v1.pth --prompt "THE RADICAL LEFT" --max_new_tokens 250
```

### 5. Run Quantitative Probes & PMI Analysis
```bash
python probe_model.py --dataset clean_truths.txt --gen_file 1000_generations.txt
```

---

## Limitations & Next Evolution (Trump-Llama)

Because TrumpGPT is an unconditional decoder-only model trained from scratch on a small corpus:
1. It **cannot follow conversational instructions** or answer direct user queries.
2. It lacks broader world knowledge outside the Truth Social dataset.

To address these limitations, we developed **[Trump-Llama-3.1-8B-Instruct](https://github.com/namgostar-tech/Trump-Llama-3.1-8B-Instruct)** as Phase 2, utilizing synthetic instruction generation via local LLMs and QLoRA fine-tuning on Meta's 8B parameter foundation model.

---

## References & Acknowledgements

- **Andrej Karpathy** for the [`nanoGPT`](https://github.com/karpathy/nanoGPT) tutorial and video lecture *"Let's build GPT: from scratch, in code, spelled out."*
- **Matt Stiles** for maintaining the [Trump Truth Social Archive](https://github.com/stiles/trump-truth-social-archive).
- **FlashAttention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) for fast exact attention.
- **Bender et al. (2021)** for *"On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜"*.

---

## Author

Developed by **Roshan Namgostar** • [namgostar.com](https://namgostar.com) • [GitHub](https://github.com/namgostar-tech)
