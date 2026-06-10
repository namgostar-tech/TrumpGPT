# TrumpGPT Architecture

The TrumpGPT model uses a decoder-only Transformer architecture, structurally similar to GPT-2. It is implemented in `train.py` using PyTorch. Below is the architecture diagram illustrating the flow of data through the model, from token inputs to output logits.

```mermaid
graph TD
    classDef model_io fill:#f9f9f9,stroke:#333,stroke-width:2px
    classDef block fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef mha fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    classDef ffn fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    %% Main Architecture Flow
    Input["Input Tokens<br>(Batch, Time)"]:::model_io --> TokEmb["Token Embedding Table<br>(vocab_size, N_EMBD)"]:::block
    PosInput["Token Positions<br>(0 to Time-1)"]:::model_io --> PosEmb["Position Embedding Table<br>(BLOCK_SIZE, N_EMBD)"]:::block
    
    TokEmb --> AddEmb(("+")):::block
    PosEmb --> AddEmb
    
    AddEmb --> Blocks["Transformer Blocks<br>(N_LAYER = 12)"]:::block
    
    Blocks --> LNF["Final LayerNorm (ln_f)"]:::block
    LNF --> LMHead["Linear LM Head<br>(N_EMBD, vocab_size)"]:::block
    LMHead --> Logits["Output Logits<br>(Batch, Time, vocab_size)"]:::model_io

    %% Detailed Transformer Block
    subgraph Transformer_Block ["Single Transformer Block (Repeated N_LAYER times)"]
        direction TB
        BIn["Block Input (x)"]:::block --> LN1["LayerNorm (ln1)"]:::block
        LN1 --> MHA["MultiHeadAttention (sa)"]:::mha
        MHA --> Add1(("+")):::block
        BIn --> Add1
        
        Add1 --> LN2["LayerNorm (ln2)"]:::block
        LN2 --> FFWD["FeedForward (ffwd)"]:::ffn
        FFWD --> Add2(("+")):::block
        Add1 --> Add2
        Add2 --> BOut["Block Output"]:::block
    end

    %% Multi-Head Attention Details
    subgraph MHA_Detail ["MultiHeadAttention (sa)"]
        direction TB
        MIn["Input"]:::mha --> CAttn["c_attn: Linear(N_EMBD, 3 * N_EMBD)"]:::mha
        CAttn --> Split["Split into Query, Key, Value<br>(N_HEAD = 12)"]:::mha
        Split --> CausalAttn["Causal Scaled Dot-Product Attention"]:::mha
        CausalAttn --> Proj["proj: Linear(N_EMBD, N_EMBD)"]:::mha
        Proj --> Drop1["Dropout"]:::mha
    end

    %% FeedForward Details
    subgraph FFWD_Detail ["FeedForward (ffwd)"]
        direction TB
        FIn["Input"]:::ffn --> L1["Linear(N_EMBD, 4 * N_EMBD)"]:::ffn
        L1 --> ReLU["ReLU Activation"]:::ffn
        ReLU --> L2["Linear(4 * N_EMBD, N_EMBD)"]:::ffn
        L2 --> Drop2["Dropout"]:::ffn
    end
```

## Model Hyperparameters

- **Vocab Size (`vocab_size`)**: 50,257 (using `gpt2` tiktoken encoding)
- **Embedding Dimension (`N_EMBD`)**: 768
- **Number of Attention Heads (`N_HEAD`)**: 12
- **Number of Transformer Blocks (`N_LAYER`)**: 12
- **Context Size/Block Size (`BLOCK_SIZE`)**: 256 tokens
- **Dropout Rate (`DROPOUT`)**: 0.2

## Components Summary

1.  **Token & Position Embeddings**: Tokens are converted into dense vectors of size 768. A learned positional embedding is added to provide the model with a sense of sequence order.
2.  **Transformer Blocks**: The model applies 12 consecutive blocks. Each block consists of:
    *   **Layer Normalization** applied *before* the multi-head attention and feed-forward networks (Pre-LN architecture).
    *   **Multi-Head Causal Self-Attention**: Projects inputs into Query, Key, and Value matrices, splitting them across 12 heads to attend to past tokens.
    *   **Feed-Forward Network**: Expands the dimensionality by a factor of 4 (768 → 3072) with a ReLU activation, and then projects it back down to 768.
    *   **Residual Connections**: The input to each sub-layer is added back to its output, preventing vanishing gradients.
3.  **Language Modeling Head**: A final Layer Normalization followed by a linear projection back to the vocabulary size to compute the probability logits for the next token.
