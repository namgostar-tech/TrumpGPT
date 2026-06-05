import torch
import argparse
import os
import train

def generate_text(prompt, weights_path, max_new_tokens):
    print(f"Using device: {train.DEVICE}")
    torch.seed()
    
    print(f"Loading model weights from {weights_path}...")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at '{weights_path}'.")
        return

    # init model using train module's classes
    model = train.BigramLanguageModel()
    m = model.to(train.DEVICE)
    
    try:
        m.load_state_dict(torch.load(weights_path, map_location=train.DEVICE))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Make sure your hyperparameter arguments (--n_embd, etc.) match the trained model.")
        return
        
    m.eval()
    print("Model loaded successfully. Generating text...\n")
    print("-" * 50)
    
    start_context = torch.tensor(train.encode(prompt), dtype=torch.long, device=train.DEVICE).unsqueeze(0)
    generated_tokens = m.generate(start_context, max_new_tokens=max_new_tokens)[0].tolist()
    print(train.decode(generated_tokens))
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using TrumpGPT.")
    parser.add_argument('--weights_path', type=str, default='TrumpGPT-Base-v1.pth')
    parser.add_argument('--prompt', type=str, default='\n')
    parser.add_argument('--max_new_tokens', type=int, default=500)
    
    # hyperparam overrides
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    
    args = parser.parse_args()
    
    # override train.py globals before initializing the model
    train.BLOCK_SIZE = args.block_size
    train.N_EMBD = args.n_embd
    train.N_HEAD = args.n_head
    train.N_LAYER = args.n_layer
    
    generate_text(args.prompt, args.weights_path, args.max_new_tokens)
