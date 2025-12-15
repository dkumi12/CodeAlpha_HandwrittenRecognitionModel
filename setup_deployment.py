import os

# 1. content for main.py
main_py_content = """from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI(title="Handwritten Character Recognition API")

# Global variables
MODEL_PATH = "mobilenet_best.h5"
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
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # We don't raise an error here so the app can still start, 
        # but predict endpoint will fail if model isn't loaded.

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # 1. Convert to Grayscale ('L')
        # 2. Resize to 128x128 using LANCZOS
        img = img.convert("L").resize((128, 128), Image.Resampling.LANCZOS)
        
        # 3. Convert back to RGB (3 channels) for MobileNetV2
        img = Image.merge("RGB", (img, img, img))
        
        # 4. Normalize to 0-1 range & add batch dimension
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)  
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image processing: {e}")

@app.get("/")
def home():
    return {"message": "Handwritten Character Recognition API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    
    predictions = model.predict(input_tensor)
    predicted_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_idx])
    
    predicted_label = LABELS[predicted_idx]
    
    return {
        "character": predicted_label,
        "confidence": confidence,
        "class_id": int(predicted_idx)
    }
"""

# 2. content for requirements.txt
requirements_txt_content = """fastapi
uvicorn
python-multipart
pillow
numpy
tensorflow-cpu
"""

# 3. content for Dockerfile
dockerfile_content = """# Use a lightweight Python 3.9 image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the model file
COPY main.py .
COPY mobilenet_best.h5 .

# Expose the port
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
"""

def create_file(filename, content):
    with open(filename, "w") as f:
        f.write(content)
    print(f"✅ Created {filename}")

if __name__ == "__main__":
    print("Generating deployment files...")
    create_file("main.py", main_py_content)
    create_file("requirements.txt", requirements_txt_content)
    create_file("Dockerfile", dockerfile_content)
    print("Done! You can now build your Docker image.")
