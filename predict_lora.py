import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Configuration
MODEL_NAME = "google/flan-t5-base"  # Base model identifier
ADAPTER_PATH = "lora_finetuned_model"  # Path to the saved LoRA adapter and tokenizer
PREFIX = "Generate UAV actions for the given BS locations: " # Prefix used during training

def main():
    parser = argparse.ArgumentParser(description="Predict UAV actions using a LoRA fine-tuned model.")
    parser.add_argument(
        "bs_locations",
        type=str,
        help="Base station locations in the format '[[x1, y1], [x2, y2], ...]', e.g., '[[100.0, 400.0]]'"
    )
    args = parser.parse_args()

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        print(f"Tokenizer loaded successfully from '{ADAPTER_PATH}'.")
    except Exception as e:
        print(f"Error loading tokenizer from '{ADAPTER_PATH}': {e}")
        print("Ensure the tokenizer was saved correctly in this directory during training.")
        return

    # Load base model
    try:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print(f"Base model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"Error loading base model '{MODEL_NAME}': {e}")
        return

    # Load LoRA PEFT model
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model = model.to(device)
        model.eval()  # Set the model to evaluation mode
        print(f"LoRA adapter loaded successfully from '{ADAPTER_PATH}' and model moved to {device}.")
    except Exception as e:
        print(f"Error loading LoRA adapter from '{ADAPTER_PATH}': {e}")
        print("Ensure the adapter_model.bin (or similar) and adapter_config.json exist in this directory.")
        return

    # Prepare input text
    # The input to the model should be in the same format as during training.
    # Based on train_lora.py, the 'input' field of the dataset was "BS Locations: [[x, y]]"
    # and then it was prepended with PREFIX.
    input_text = f"{PREFIX}BS Locations: {args.bs_locations}"
    print(f"Input to model: {input_text}")

    # Tokenize input
    try:
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to(device)
    except Exception as e:
        print(f"Error tokenizing input: {e}")
        return
        
    # Generate output
    print("Generating actions...")
    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,       # As requested
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=5,  # Using beam search
                do_sample=False, # Disable sampling
                repetition_penalty=1.2 # Keep repetition penalty
                # encoder_no_repeat_ngram_size is removed
                # top_k and top_p are removed as they are for sampling
            )
        
        # Decode the generated tokens to text
        # The [0] is to access the first (and only, in this case) sequence in the batch
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True) # Revert to True
        
        print(f"Predicted actions: {decoded_output}")

    except Exception as e:
        print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    main() 