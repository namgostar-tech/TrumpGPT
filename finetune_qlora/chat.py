import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, TextStreamer
import transformers
transformers.logging.set_verbosity_error()
from peft import PeftModel

# 1. Configuration
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
adapter_path = "./results/final_adapter"

print("Loading 4-bit quantization config...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. Load Base Model and Tokenizer
print(f"Loading base model: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa" if torch.cuda.is_bf16_supported() else "eager"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 3. Load the LoRA Adapter
print(f"Loading adapter from {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)

# 4. Create a generation pipeline and streamer
print("Ready for chatting! Type 'quit' to exit.\n")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 5. Interactive Loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'stop']:
        break

    # Format the input using the model's chat template
    messages = [
        {"role": "system", "content": "You are Donald Trump. Answer the user's questions in your unique voice, tone, and style. Dive straight into your response without repeating the question."},
        {"role": "user", "content": user_input}
    ]
    
    # Define when the model should stop generating
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Generate response
    print("\nTrumpGPT: ", end="", flush=True)
    outputs = pipe(
        messages,
        max_new_tokens=512,
        min_new_tokens=25,
        max_length=None, # Explicitly remove default to suppress the warning
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=terminators, # Stop generating when it outputs an end-of-turn token
        streamer=streamer
    )
    
    print("\n" + "-" * 50)
