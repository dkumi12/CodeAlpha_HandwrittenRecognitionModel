"""
Handwritten Character Recognition API

A FastAPI backend service for recognizing handwritten characters (A-Z, a-z, 0-9)
using a MobileNetV2-based CNN model.

Performance:
    - Top-1 Accuracy: 75.37%
    - Top-3 Accuracy: 94.28%
    - 62 character classes

Author: Daniel Kumi
Version: 1.0.0
"""

import os
import io
from typing import Dict, Any, Optional, List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

# ==========================================
# Application Configuration
# ==========================================

app = FastAPI(
    title="Handwritten Character Recognition API",
    description="Recognize handwritten characters (A-Z, a-z, 0-9) using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Model Configuration
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH: str = os.path.join(BASE_DIR, "model", "mobilenet_best.h5")
INPUT_SIZE: tuple = (128, 128)
NUM_CLASSES: int = 62

# Global model instance
model: Optional[tf.keras.Model] = None

# Class labels: 0-9, A-Z, a-z (62 classes total)
LABELS: List[str] = sorted(
    [str(i) for i in range(10)] +
    [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
    [chr(i) for i in range(ord('a'), ord('z') + 1)]
)


# ==========================================
# Model Loading
# ==========================================

@app.on_event("startup")
def load_model() -> None:
    """
    Load the trained model on application startup.
    
    This function is called automatically when the FastAPI server starts.
    It loads the MobileNetV2-based model from the specified path.
    
    Raises:
        Prints error message if model file not found or loading fails.
    """
    global model
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Error: Model file not found at {MODEL_PATH}")
            print("   Ensure the model is downloaded during Docker build.")
            return

        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output classes: {NUM_CLASSES}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")


# ==========================================
# Image Preprocessing
# ==========================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes for model inference.
    
    This function takes raw image bytes (from an upload), converts them
    to the format expected by the MobileNetV2 model:
    - RGB color mode (3 channels)
    - 128x128 pixel resolution
    - Normalized to [0, 1] range
    - Batch dimension added
    
    Args:
        image_bytes: Raw bytes of the uploaded image file.
        
    Returns:
        np.ndarray: Preprocessed image array of shape (1, 128, 128, 3)
        
    Raises:
        HTTPException: If image processing fails (invalid format, corrupted, etc.)
        
    Example:
        >>> img_array = preprocess_image(uploaded_file.read())
        >>> print(img_array.shape)
        (1, 128, 128, 3)
    """
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB (MobileNetV2 expects 3 channels)
        # This handles grayscale, RGBA, and other formats
        img = img.convert('RGB')

        # Resize to model's expected input size
        img = img.resize(INPUT_SIZE, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {str(e)}. Please upload a valid image file."
        )


# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
def root() -> Dict[str, str]:
    """
    Root endpoint - API health check.
    
    Returns:
        Dict with welcome message and status.
    """
    return {
        "message": "Handwritten Character Recognition API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.
    
    Returns:
        Dict containing model status and configuration.
    """
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "num_classes": NUM_CLASSES,
        "input_size": INPUT_SIZE
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict the character in an uploaded image.
    
    This endpoint accepts an image file containing a handwritten character
    and returns the model's prediction along with confidence score.
    
    Args:
        file: Uploaded image file (PNG, JPEG, etc.)
        
    Returns:
        Dict containing:
            - prediction: The predicted character (str)
            - confidence: Confidence score between 0 and 1 (float)
            - class_id: Numeric class index (int)
            
    Raises:
        HTTPException 503: If model is not loaded
        HTTPException 400: If image processing fails
        
    Example Response:
        {
            "prediction": "A",
            "confidence": 0.9823,
            "class_id": 10
        }
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Read and preprocess image
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    
    # Run inference
    predictions = model.predict(processed_image, verbose=0)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    
    # Map index to label
    predicted_label = LABELS[predicted_index] if predicted_index < len(LABELS) else "Unknown"

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "class_id": predicted_index
    }


@app.post("/predict/top_k")
async def predict_top_k(
    file: UploadFile = File(...),
    k: int = 3
) -> Dict[str, Any]:
    """
    Get top-K predictions for an uploaded image.
    
    Useful for understanding model confidence across multiple classes,
    especially for visually similar characters (O/0, I/1, l/1).
    
    Args:
        file: Uploaded image file
        k: Number of top predictions to return (default: 3)
        
    Returns:
        Dict containing list of top-K predictions with labels and scores.
        
    Example Response:
        {
            "top_predictions": [
                {"label": "O", "confidence": 0.45, "class_id": 24},
                {"label": "0", "confidence": 0.35, "class_id": 0},
                {"label": "Q", "confidence": 0.10, "class_id": 26}
            ]
        }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Limit k to reasonable range
    k = max(1, min(k, NUM_CLASSES))
    
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    
    predictions = model.predict(processed_image, verbose=0)[0]
    
    # Get top-K indices and scores
    top_indices = np.argsort(predictions)[-k:][::-1]
    
    top_predictions = [
        {
            "label": LABELS[idx] if idx < len(LABELS) else "Unknown",
            "confidence": round(float(predictions[idx]), 4),
            "class_id": int(idx)
        }
        for idx in top_indices
    ]
    
    return {"top_predictions": top_predictions}
