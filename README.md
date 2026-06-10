# TrumpGPT

If you are a sane and compassionate person watching nations scramble for superiority in an AI arms race, you might be morbidly curious why someone would take a stab at training their own language model… on Donald Trump’s Truth Social profile. 

This project explores the intersection of political rhetoric and generative AI. It sheds light on exactly what certain authoritarian figures and language models have in common: the impressive ability to execute style mimicry and persuasion, while frequently falling short of logical comprehension and semantic reasoning.

This repository features two distinct models developed to explore this phenomenon:

## TrumpGPT-Base-v1

The first iteration of TrumpGPT is a decoder-only architecture trained from scratch to unconditionally generate text mirroring Trump's style.

### Architecture
- **Base**: Built upon Andrej Karpathy's `nanoGPT` with significant capacity scale-ups.
- **Tokenization**: OpenAI’s `tiktoken` (`gpt2` subword encoding) instead of character-level tokenization.
- **Scale**: Upgraded embedding dimension (`n_embd`) to 768 and context window (`block_size`) to 256.
- **Layers & Heads**: Scaled up to 12 attention heads (`n_head`) and 12 layers (`n_layer`).
- **Attention**: Replaced manual attention implementation with PyTorch's `F.scaled_dot_product_attention(..., is_causal=True)` for FlashAttention optimization, speeding up training and reducing memory usage.
- **Regularization**: Dropout rate of 0.2 and standard "pre-norm" LayerNorm.

### Training Details
- **Dataset**: Built from a June 4th, 2026 snapshot of Matt Stiles' "Trump Truth Social Archive". The dataset was heavily cleaned (removing URLs, re-truths, empty bodies, and collapsing formatting) and appended with `<|endoftext|>` tokens, resulting in 11,583 viable training posts.
- **Optimization**: Utilized PyTorch's `autocast` (`bfloat16`/`float16`) and set `torch.set_float32_matmul_precision('high')`.
- **Scheduler**: Custom learning rate scheduler with a warmup phase and cosine decay, accompanied by gradient clipping at 1.0 to prevent exploding gradients.
- **Checkpointing**: Automated validation loss tracking with early stopping (patience of 5 evaluations) to prevent overfitting.
- **Outcome**: The model settled at a validation loss of 1.412 around 1,500 training iterations, resulting in a ~650MB model weight file.

### Analysis & Results
The model successfully mirrors the rhetorical aesthetic of its training data. Stylistic metric analysis showed it accurately captured high capitalization ratios (20.01% of words) and exclamation frequency. PMI (Pointwise Mutual Information) analysis revealed it perfectly memorized polarized vocabulary associations (tethering "enemies" to nicknames and "allies" to words revolving around conflict/victory) but completely failed to retain any underlying specific semantic meaning. It acts as a "stochastic parrot", relying on sensationalist rhetoric to generate word salads devoid of logical consistency.

---
*By Roshan Namgostar • June 2026*
