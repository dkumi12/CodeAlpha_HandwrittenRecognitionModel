import streamlit as st
import requests
from PIL import Image
import io

# --- CONFIGURATION ---
# If running locally, put your EC2 Public IP here (e.g., "http://54.123.45.67:8000")
# If running on the same server, use "http://localhost:8000"
API_URL = "http://localhost:8000/predict" 

st.set_page_config(page_title="Handwritten Recognition", layout="centered")

st.title("✍️ Handwritten Character AI")
st.write("Upload an image of a handwritten digit or character to identify it!")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Upload', width=300)
    
    # 2. Convert to bytes for API
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    # 3. Predict Button
    if st.button("🔍 Identify Character"):
        with st.spinner("Analyzing..."):
            try:
                # Send request to your FastAPI backend
                files = {"file": ("image.png", byte_im, "image/png")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display Results nicely
                    st.success(f"**Prediction:** {result['prediction']}")
                    st.info(f"**Confidence:** {result['confidence']:.2%}")
                    
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the API. Is the Docker container running?")
