import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 8B using QLoRA")
    parser.add_argument("--dataset", type=str, default="qa_dataset.jsonl", help="Path to the JSONL Q&A dataset")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    # 1. Load Dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset('json', data_files={'train': args.dataset}, split='train')
    
    # 2. BitsAndBytes Configuration for 4-bit Quantization
    # This is critical for fitting an 8B model on a 12GB RTX 4070
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16

    # 4. Load Base Model
    print("Loading base model in 4-bit precision...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto", # Automatically assigns to GPU
        attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else "eager"
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Configuration
    # We only train adapters, keeping base weights frozen
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Training Arguments
    # These are highly optimized for 12GB VRAM
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,     # Very small batch size to fit in VRAM
        gradient_accumulation_steps=8,     # Accumulate gradients to simulate batch_size=8
        optim="paged_adamw_32bit",         # Memory efficient optimizer
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(), # Use bf16 if supported by hardware (RTX 4070 supports it)
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none" # Disable wandb/tensorboard for simplicity unless requested
    )

    # 7. SFTTrainer
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="messages", # trl automatically handles chat templates if structured correctly
        max_seq_length=2048,           # Control sequence length to manage VRAM
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # Apply the chat template to the messages column
    def format_chat_template(row):
        row["messages"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return row
        
    dataset = dataset.map(format_chat_template, num_proc=4)
    trainer.train_dataset = dataset
    trainer.dataset_text_field = "messages"

    # 8. Train!
    print("Starting training...")
    trainer.train()

    # 9. Save final adapter
    print(f"Saving final model adapter to {args.output_dir}/final_adapter")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    print("Done!")

if __name__ == "__main__":
    main()
