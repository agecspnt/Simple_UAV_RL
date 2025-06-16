import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
MODEL_NAME = "google/flan-t5-base"
DATASET_PATH = "converted_dataset.jsonl" # Input data file
OUTPUT_DIR = "lora_finetuned_model"     # Output directory for the model and tokenizer

# LoRA parameters (as specified by user)
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q", "v"] 
TASK_TYPE = TaskType.SEQ_2_SEQ_LM

# Training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 2e-5 # Adjusted learning rate

# Tokenization parameters
# Max sequence lengths. Adjust based on your data.
# Input: "BS Locations: [[54.7, 8.2]]" -> Relatively short
# Output: "[1,1,3], [2,0,3], [3,1,2], ..." -> Can be longer
MAX_SOURCE_LENGTH = 128 
MAX_TARGET_LENGTH = 256 
# Prefix for T5 model input, important for guiding the model
PREFIX = "Generate UAV actions for the given BS locations: "

def main():
    # Check for GPU availability for fp16
    fp16_enabled = False # Temporarily disable fp16 for debugging
    if torch.cuda.is_available():
        print("CUDA is available. FP16 training will be manually disabled for debugging.")
    else:
        print("CUDA is not available. FP16 training will be disabled. Training will run on CPU.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Preprocessing function
    def preprocess_function(examples):
        inputs = [PREFIX + doc for doc in examples["input"]]
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs, 
            max_length=MAX_SOURCE_LENGTH, 
            truncation=True, 
            padding="max_length" # Pad to max_length
        )

        # Tokenize targets (outputs)
        # The tokenizer.as_target_tokenizer() context manager is used for seq2seq models
        # to ensure outputs are tokenized correctly for the decoder.
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["output"], 
                max_length=MAX_TARGET_LENGTH, 
                truncation=True, 
                padding="max_length" # Pad to max_length
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Load dataset
    try:
        # Assumes 'converted_dataset.jsonl' is in the same directory or path is correctly specified
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train") 
    except Exception as e:
        print(f"Error loading dataset from '{DATASET_PATH}': {e}")
        print("Please ensure 'converted_dataset.jsonl' exists in the current directory or the path is correct,")
        print("and it is a valid JSON Lines file with 'input' and 'output' fields per line.")
        return

    # Validate dataset columns
    if "input" not in dataset.column_names or "output" not in dataset.column_names:
        print(f"Dataset at '{DATASET_PATH}' must contain 'input' and 'output' columns.")
        print(f"Found columns: {dataset.column_names}")
        return
        
    print(f"Dataset loaded successfully. Number of examples: {len(dataset)}")
    print(f"Example entry: Input: {dataset[0]['input']}, Output: {dataset[0]['output']}")

    # Apply preprocessing to the dataset
    # batched=True processes multiple elements of the dataset at once for efficiency
    # remove_columns removes the original text columns after tokenization
    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    print("Dataset preprocessing complete.")

    # Load base model (flan-t5-base)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print(f"Base model '{MODEL_NAME}' loaded.")

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none", # Common practice for LoRA, alternatives: "all", "lora_only"
        task_type=TASK_TYPE
    )

    # Apply LoRA to the model using PEFT
    model = get_peft_model(model, lora_config)
    print("LoRA configured and applied to the model.")
    model.print_trainable_parameters() # Shows the percentage of parameters being trained

    # Training arguments for Seq2Seq model
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        # per_device_eval_batch_size=BATCH_SIZE, # No evaluation dataset specified in this script
        learning_rate=LEARNING_RATE,
        weight_decay=0.01, # Standard weight decay
        fp16=fp16_enabled, # Enable mixed-precision training if GPU is available
        logging_dir=os.path.join(OUTPUT_DIR, "logs"), # Directory for TensorBoard logs
        logging_steps=max(1, len(tokenized_dataset) // (BATCH_SIZE * 10) if len(tokenized_dataset) > BATCH_SIZE * 10 else 10), # Log ~10 times per epoch or every 10 steps
        save_strategy="epoch", # Save model checkpoint at the end of each epoch
        save_total_limit=2, # Optional: limit the number of checkpoints saved
        # load_best_model_at_end=True, # Optional: if using evaluation
        # metric_for_best_model="eval_loss", # Optional: if using evaluation
        # greater_is_better=False, # Optional: if using evaluation
        report_to="tensorboard", # Logs metrics to TensorBoard
        max_grad_norm=1.0, # Added gradient clipping
        label_names=["labels"], # Explicitly tell the Trainer the name of the label column
    )

    # Data collator for Seq2Seq tasks
    # This will handle padding of inputs and labels dynamically per batch.
    # It also prepares decoder_input_ids and replaces padding token ID in labels with -100.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id, # Use pad_token_id for label padding
        pad_to_multiple_of=8 if fp16_enabled else None # Optimization for fp16
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # eval_dataset=None, # No evaluation dataset specified
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    print(f"Starting LoRA fine-tuning for {NUM_EPOCHS} epochs...")
    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Ensure your data is correctly formatted and batch size is appropriate for your hardware.")
        return

    # Save the LoRA model adapter and tokenizer
    # This saves only the LoRA adapter weights, not the full base model.
    # To load this later:
    # from peft import PeftModel
    # base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    print(f"Saving LoRA adapter and tokenizer to '{OUTPUT_DIR}'...")
    try:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Training complete. LoRA adapter and tokenizer saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")


if __name__ == "__main__":
    main()