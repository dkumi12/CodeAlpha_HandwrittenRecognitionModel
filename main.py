import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

app = FastAPI(title="Handwritten Character Recognition API")

# ==========================================
# 1. SETUP PATHS (Modified)
# ==========================================
# Get the directory where main.py is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to the 'model' folder inside the base directory
MODEL_PATH = os.path.join(BASE_DIR, "model", "mobilenet_best.h5")

model = None

# Hardcoded class mapping (Digits 0-9, Uppercase A-Z, Lowercase a-z)
LABELS = sorted(
    [str(i) for i in range(10)] +
    [chr(i) for i in range(ord('A'), ord('Z')+1)] +
    [chr(i) for i in range(ord('a'), ord('z')+1)]
)

@app.on_event("startup")
def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Error: Model file not found at {MODEL_PATH}")
            return

        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def preprocess_image(image_bytes):
    """
    Reads image bytes, converts to grayscale/RGB, resizes, and normalizes.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # 1. Convert to RGB (MobileNet typically expects 3 channels)
        # Note: Your snippet said 'Grayscale', but standard MobileNet is RGB. 
        # If your model was trained on Grayscale, change 'RGB' to 'L'.
        img = img.convert('RGB') 

        # 2. Resize to the input shape expected by your model
        # Standard MobileNet uses 224x224. Change this if your model is different.
        img = img.resize((128, 128))

        # 3. Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]

        # 4. Add batch dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image processing: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read file
    image_bytes = await file.read()
    
    # Preprocess
    processed_image = preprocess_image(image_bytes)
    
    # Predict
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    
    predicted_label = LABELS[predicted_index] if predicted_index < len(LABELS) else "Unknown"

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "class_id": int(predicted_index)
    }
