"""Offline inference using local model."""

from transformers import pipeline

MODEL_PATH = "./model" 

def setup_classifier():
    """Setup the classifier with local model, forcing CPU usage."""
    print("Setting up classifier on CPU...")
    
    # Force CPU
    device = -1  # Always use CPU
    
    classifier = pipeline("text-classification", model="./model", tokenizer="./model")
    return classifier

def predict(texts, classifier=None):
    """Make predictions on texts."""
    if classifier is None:
        classifier = setup_classifier()
    
    results = classifier(texts, top_k=None, function_to_apply="sigmoid")
    return results

if __name__ == "__main__":
    # Example usage
    texts = [
        "This is so much fun!",
        "I want to go back home to my dogs, i'll be happy to go back to them."
    ]
    
    classifier = setup_classifier()
    results = predict(texts, classifier)
    
    for text, result in zip(texts, results):
        print(f"\nText: {text}")
        print("Emotions:")
        for emotion in result:
            if emotion['score'] > 0.5:  # Only show emotions with >50% confidence
                print(f"- {emotion['label']}: {emotion['score']:.2%}") 