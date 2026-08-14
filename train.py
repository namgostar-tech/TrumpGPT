import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import argparse
import tiktoken
import os
import contextlib

# -----------------------------------------------------------------------------
# Default Hyperparameters & Architecture Configuration
# Comparable to GPT-2 (small) architecture scaled up from nanoGPT
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS = 10000
EVAL_INTERVAL = 250
LEARNING_RATE = 3e-4
EVAL_ITERS = 50
DATA_FILE_PATH = 'clean_truths.txt'

N_EMBD = 768
N_HEAD = 12
N_LAYER = 12
DROPOUT = 0.2

# Learning Rate Scheduler
WARMUP_ITERS = 200
LR_DECAY_ITERS = 10000
MIN_LR = 3e-5

# Gradient Clipping
GRAD_CLIP = 1.0

# Device Selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_autocast_ctx():
    """Returns an appropriate autocast context for the active accelerator."""
    if DEVICE.type == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type='cuda', dtype=dtype)
    elif DEVICE.type == 'cpu':
        return torch.autocast(device_type='cpu', dtype=torch.bfloat16)
    else:
        try:
            return torch.autocast(device_type='mps', dtype=torch.float16)
        except Exception:
            return contextlib.nullcontext()

# Tokenizer (tiktoken gpt2 subword encoding)
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# -----------------------------------------------------------------------------
# Data Loader Helpers
# -----------------------------------------------------------------------------

def get_batch(split, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,), device=data.device)
    grid = ix.unsqueeze(1) + torch.arange(BLOCK_SIZE, device=data.device)
    x = data[grid]
    y = data[grid + 1]
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            with get_autocast_ctx():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    """Cosine learning rate decay with linear warmup."""
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

# -----------------------------------------------------------------------------
# Model Architecture (Scaled Decoder-Only Transformer with FlashAttention)
# -----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        assert N_EMBD == num_heads * head_size
        self.num_heads = num_heads
        self.head_size = head_size
        # Batch all query, key, value projections for all heads
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate Q, K, V for all heads and transpose for batch attention
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N_EMBD, dim=2)
        
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        # Causal scaled dot-product attention (utilizes FlashAttention / memory-efficient kernels)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-norm formulation with residual skip connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == enc.eot_token:
                break
        return idx

# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or generate text with TrumpGPT.")
    parser.add_argument('--mode', choices=['train', 'generate'], required=True, help="Mode: 'train' or 'generate'")
    parser.add_argument('--weights_path', type=str, default='TrumpGPT-Base-v1.pth', help="Path to model weights file")
    parser.add_argument('--max_new_tokens', type=int, default=500, help="Max tokens to generate in generate mode")
    parser.add_argument('--prompt', type=str, default='\n', help="Seed prompt for generation")
    
    # Hyperparameter overrides
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--block_size', type=int, default=256, help="Context window length")
    parser.add_argument('--n_embd', type=int, default=768, help="Embedding dimension")
    parser.add_argument('--n_head', type=int, default=12, help="Number of attention heads")
    parser.add_argument('--n_layer', type=int, default=12, help="Number of transformer layers")
    parser.add_argument('--max_iters', type=int, default=10000, help="Maximum training iterations")
    parser.add_argument('--data_file', type=str, default=DATA_FILE_PATH, help="Path to clean training text file")
    
    args = parser.parse_args()

    # Override module globals with CLI args
    BATCH_SIZE = args.batch_size
    BLOCK_SIZE = args.block_size
    N_EMBD = args.n_embd
    N_HEAD = args.n_head
    N_LAYER = args.n_layer
    MAX_ITERS = args.max_iters
    DATA_FILE_PATH = args.data_file

    # Seed
    torch.manual_seed(1337)
    if DEVICE.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    # Initialize model
    model = BigramLanguageModel()
    m = model.to(DEVICE)

    if args.mode == 'train':
        print(f"Using device: {DEVICE}")
        if DEVICE.type == 'cuda':
            print(f"GPU Model: {torch.cuda.get_device_name(0)}")

        if not os.path.exists(DATA_FILE_PATH):
            print(f"Error: The training data file '{DATA_FILE_PATH}' was not found.")
            print("Please run 'python extract_clean_truths.py' first to prepare the dataset.")
            exit(1)

        print(f"Loading training data from {DATA_FILE_PATH}...")
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            text = f.read()

        data = torch.tensor(encode(text), dtype=torch.long, device=DEVICE)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

        param_count = sum(p.numel() for p in m.parameters())
        print(f"Model capacity: {param_count/1e6:.2f}M parameters")

        optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

        print("\nStarting training loop...")
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for iter in range(MAX_ITERS):
            lr = get_lr(iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
                losses = estimate_loss(m, train_data, val_data)
                print(f"\nStep {iter:>5}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")
                
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    patience_counter = 0
                    print(f"--> Best validation loss reached ({best_val_loss:.4f})! Saving weights to {args.weights_path}...")
                    torch.save(m.state_dict(), args.weights_path)
                else:
                    patience_counter += 1
                    print(f"--> Validation loss did not improve. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered! Training stopped to prevent overfitting.")
                        break
            elif iter % 10 == 0:
                print(".", end="", flush=True)

            xb, yb = get_batch('train', train_data, val_data)
            with get_autocast_ctx():
                logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
            optimizer.step()

        print("\nTraining complete. Best model weights saved.")

    elif args.mode == 'generate':
        print(f"Using device: {DEVICE}")
        torch.seed()

        print(f"Loading model weights from '{args.weights_path}'...")
        if not os.path.exists(args.weights_path):
            print(f"Error: Weights file not found at '{args.weights_path}'. Please train the model first.")
            exit(1)

        try:
            m.load_state_dict(torch.load(args.weights_path, map_location=DEVICE))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Ensure architecture arguments (--n_embd, --n_layer, --n_head, etc.) match the trained checkpoint.")
            exit(1)

        m.eval()
        print("Model loaded successfully. Generating text...\n")
        print("=" * 60)
        start_context = torch.tensor(encode(args.prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
        generated_tokens = m.generate(start_context, max_new_tokens=args.max_new_tokens)[0].tolist()
        print(decode(generated_tokens))
        print("=" * 60)
