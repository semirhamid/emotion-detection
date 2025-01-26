import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "lrei/roberta-large-emolit"
OUTPUT_DIR = "./model"

def print_model_folder_contents(folder_path):
    """Print the contents of the specified folder."""
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    
    print(f"\nContents of folder '{folder_path}':")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            print(f"- {os.path.join(root, file)}")

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
    print_model_folder_contents(OUTPUT_DIR)  # Print folder contents

if __name__ == "__main__":
    download_model()
