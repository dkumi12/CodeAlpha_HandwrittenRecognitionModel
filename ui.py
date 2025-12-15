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

# Initialize session state for the canvas key (this is the trick to clear it!)
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = "canvas_v1"

# Create Tabs
tab1, tab2 = st.tabs(["🖌️ Draw Here", "📂 Upload File"])
image_to_send = None

# --- TAB 1: DRAWING CANVAS ---
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Draw a digit or letter below:")
        # Create the drawing pad with a dynamic key
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=15,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            # We use the session state key here. Changing this variable clears the canvas!
            key=st.session_state.canvas_key,
        )

    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        # The Clear Button
        if st.button("🗑️ Clear"):
            # We change the key, forcing Streamlit to re-render a fresh empty canvas
            st.session_state.canvas_key = str(int(st.session_state.canvas_key.split("_v")[1]) + 1) if "_v" in st.session_state.canvas_key else "canvas_v1"
            st.rerun()

    if canvas_result.image_data is not None:
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
                if image_to_send.mode != "RGB":
                    image_to_send = image_to_send.convert("RGB")
                
                buf = io.BytesIO()
                image_to_send.save(buf, format="PNG")
                byte_im = buf.getvalue()

                files = {"file": ("image.png", byte_im, "image/png")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Prediction:** {result['prediction']}")
                    st.info(f"**Confidence:** {result['confidence']:.2%}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
