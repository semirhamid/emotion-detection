"""FastAPI server for emotion classification."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from inference import setup_classifier, predict

app = FastAPI(
    title="Emotion Classification API",
    description="API for multi-label emotion classification",
    version="1.0.0"
)

# Initialize the model
classifier = setup_classifier()

class TextRequest(BaseModel):
    texts: List[str]
    threshold: float = 0.5

class EmotionResponse(BaseModel):
    text: str
    emotions: Dict[str, float]

@app.post("/predict", response_model=List[EmotionResponse])
async def predict_emotions(request: TextRequest):
    """Predict emotions for a list of texts."""
    try:
        # Get predictions
        results = predict(request.texts, classifier)
        
        # Format response
        responses = []
        for text, result in zip(request.texts, results):
            # Filter emotions above threshold
            emotions = {
                emotion["label"]: emotion["score"]
                for emotion in result
                if emotion["score"] >= request.threshold
            }
            responses.append(EmotionResponse(text=text, emotions=emotions))
        
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Emotion Classification API For NLP Project",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict emotions for a list of texts",
            "/": "GET - This information"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
