"""Download model from Hugging Face and save locally."""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "lrei/roberta-large-emolit"
OUTPUT_DIR = "./model"

def download_model():
    """Download and save the model locally."""
    print(f"Downloading model {MODEL_NAME}...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download and save tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Download and save model
    print("Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(OUTPUT_DIR)
    
    print(f"\nModel and tokenizer saved to {OUTPUT_DIR}")
    print("You can now use the local model with inference.py")

if __name__ == "__main__":
    download_model() 