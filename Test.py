import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle

# Load model and label binarizer
try:
    model = load_model('handwritten_character_recognition_model.h5')
    with open('label_binarizer.pkl', 'rb') as f:
        lb = pickle.load(f)
    classes = lb.classes_
    st.write(f"Loaded {len(classes)} classes: {classes}")
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()

# Streamlit app
st.title("Handwritten Character Recognition")
st.write("Upload a 28x28 grayscale image of a handwritten character (0-9, A-Z, a-z).")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = image.convert('L')  # Grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model
        st.write(f"Processed image shape: {img_array.shape}")

        # Predict
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display result
        st.write(f"Predicted Character: **{predicted_class}** (Confidence: {confidence:.2f})")
    except Exception as e:
        st.error(f"Error processing image or predicting: {e}")