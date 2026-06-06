import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 using QLoRA")
    parser.add_argument("--dataset", type=str, default="qa_dataset.jsonl", help="Path to the JSONL Q&A dataset")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    # 1. Load Dataset and Create Validation Split
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset('json', data_files={'train': args.dataset}, split='train')
    
    # 5% val split to watch for overfitting
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    # 2. bnb config for 4-bit quant
    # crucial for fitting 8b/12b on the 12gb rtx 4070
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
    tokenizer.padding_side = "right" # fix for fp16

    # 4. Load Base Model
    print("Loading base model in 4-bit precision...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto", # auto pick gpu
        attn_implementation="sdpa" if torch.cuda.is_bf16_supported() else "eager"
    )
    
    # turn on grad checkpointing to save vram
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Configuration
    peft_config = LoraConfig(
        r=8,               # reduced rank to save VRAM
        lora_alpha=16,     # scaled alpha accordingly
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,  # bumped to stop overfitting
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # SFTTrainer will automatically apply the PEFT config to the base model.
    # We no longer need to manually wrap it with get_peft_model.

    # 6. Training Arguments
    training_arguments = SFTConfig(
        max_length=1024,               # increased to 1024; fits comfortably in 12GB VRAM
        output_dir=args.output_dir,
        num_train_epochs=1,                # decreased, 1-2 epochs usually good for fine-tuning
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="steps",             # check eval occasionally
        eval_steps=50,                     
        optim="paged_adamw_8bit",          # switched to 8-bit to save VRAM
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.05,                 # bumped up to stop memorization
        load_best_model_at_end=True,       # need this for early stopping
        metric_for_best_model="eval_loss", 

        fp16=False,
        bf16=torch.cuda.is_bf16_supported(), # use bf16 if my hardware supports it
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none" # turn off wandb for now
    )

    # 7. SFTTrainer
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], # abort if eval loss goes up twice
    )

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
