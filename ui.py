import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Handwritten Character AI", layout="centered")

st.title("✍️ Handwritten Character AI")
st.write("Draw a character or upload an image to identify it!")

# Create Tabs
tab1, tab2 = st.tabs(["🖌️ Draw Here", "📂 Upload File"])
image_to_send = None

# --- TAB 1: DRAWING CANVAS ---
with tab1:
    st.write("Draw a digit or letter below:")
    
    # Create the drawing pad
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill
        stroke_width=15,                      # Thick brush
        stroke_color="#000000",               # Black ink
        background_color="#FFFFFF",           # White paper
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert canvas data to image
        img_array = canvas_result.image_data.astype('uint8')
        image_to_send = Image.fromarray(img_array)

# --- TAB 2: FILE UPLOADER ---
with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_to_send = Image.open(uploaded_file)
        st.image(image_to_send, caption='Your Upload', width=200)

# --- PREDICTION BUTTON ---
if st.button("🔍 Identify Character"):
    if image_to_send is None:
        st.warning("⚠️ Please draw something or upload an image first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Convert to RGB (standardize input)
                if image_to_send.mode != "RGB":
                    image_to_send = image_to_send.convert("RGB")
                
                # Convert to Bytes
                buf = io.BytesIO()
                image_to_send.save(buf, format="PNG")
                byte_im = buf.getvalue()

                # Send to API
                files = {"file": ("image.png", byte_im, "image/png")}
                response = requests.post(API_URL, files=files)
                
                # Show Result
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Prediction:** {result['prediction']}")
                    st.info(f"**Confidence:** {result['confidence']:.2%}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
