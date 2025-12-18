"""
Handwritten Character Recognition - Streamlit UI

An interactive web interface for the handwritten character recognition model.
Users can draw characters on a canvas or upload images for recognition.

Features:
    - Drawing canvas with adjustable stroke
    - Image file upload support
    - Real-time prediction display
    - Confidence score visualization

Author: Daniel Kumi
Version: 1.0.0
"""

import io
from typing import Optional, Dict, Any

import streamlit as st
import requests
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# ==========================================
# Configuration
# ==========================================

API_URL: str = "http://localhost:8000/predict"
CANVAS_SIZE: int = 280
STROKE_WIDTH: int = 15

# ==========================================
# Page Setup
# ==========================================

st.set_page_config(
    page_title="Handwritten Character AI",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# Session State Initialization
# ==========================================

if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = "canvas_v1"


# ==========================================
# Helper Functions
# ==========================================

def increment_canvas_key() -> None:
    """
    Increment the canvas key to force a re-render (clear canvas).
    
    Streamlit's drawable canvas doesn't have a native clear function,
    so we change the component key to force it to re-initialize.
    """
    current_key = st.session_state.canvas_key
    if "_v" in current_key:
        version = int(current_key.split("_v")[1])
        st.session_state.canvas_key = f"canvas_v{version + 1}"
    else:
        st.session_state.canvas_key = "canvas_v1"


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB mode if necessary.
    
    Args:
        image: PIL Image in any color mode
        
    Returns:
        PIL Image in RGB mode
    """
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def send_prediction_request(image: Image.Image) -> Dict[str, Any]:
    """
    Send image to the prediction API and return results.
    
    Args:
        image: PIL Image to classify
        
    Returns:
        Dictionary containing prediction results or error info
        
    Raises:
        requests.exceptions.RequestException: If API request fails
    """
    # Convert to RGB and save to buffer
    rgb_image = convert_to_rgb(image)
    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    byte_data = buf.getvalue()
    
    # Send to API
    files = {"file": ("image.png", byte_data, "image/png")}
    response = requests.post(API_URL, files=files, timeout=30)
    
    if response.status_code == 200:
        return {"success": True, "data": response.json()}
    else:
        return {
            "success": False,
            "error": f"Error {response.status_code}: {response.text}"
        }


# ==========================================
# Main UI
# ==========================================

def main() -> None:
    """Main application entry point."""
    
    # Header
    st.title("✍️ Handwritten Character AI")
    st.markdown(
        "Draw a character or upload an image to identify it! "
        "Supports **A-Z**, **a-z**, and **0-9**."
    )
    
    # Model info expander
    with st.expander("ℹ️ About this model"):
        st.markdown("""
        **Architecture:** MobileNetV2 (Transfer Learning)
        
        **Performance:**
        - Top-1 Accuracy: 75.37%
        - Top-3 Accuracy: 94.28%
        
        **Known Limitations:**
        - May confuse similar characters (O/0, I/1, l/1)
        - Works best with clear, centered characters
        """)
    
    st.divider()
    
    # Initialize image container
    image_to_send: Optional[Image.Image] = None
    
    # Create tabs for input methods
    tab_draw, tab_upload = st.tabs(["🖌️ Draw Here", "📂 Upload File"])
    
    # ==========================================
    # Tab 1: Drawing Canvas
    # ==========================================
    with tab_draw:
        col_canvas, col_controls = st.columns([3, 1])
        
        with col_canvas:
            st.caption("Draw a digit or letter below:")
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=STROKE_WIDTH,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=CANVAS_SIZE,
                width=CANVAS_SIZE,
                drawing_mode="freedraw",
                key=st.session_state.canvas_key,
                display_toolbar=True
            )
        
        with col_controls:
            st.write("")  # Spacer
            st.write("")  # Spacer
            
            if st.button("🗑️ Clear", use_container_width=True):
                increment_canvas_key()
                st.rerun()
            
            st.caption("Tips:")
            st.caption("• Draw large")
            st.caption("• Center character")
            st.caption("• Use clear strokes")
        
        # Extract image from canvas if drawing exists
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.astype('uint8')
            # Check if canvas has any drawing (not all white)
            if np.any(img_array[:, :, :3] != 255):
                image_to_send = Image.fromarray(img_array)
    
    # ==========================================
    # Tab 2: File Upload
    # ==========================================
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "gif", "bmp"],
            help="Upload an image containing a handwritten character"
        )
        
        if uploaded_file is not None:
            image_to_send = Image.open(uploaded_file)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    image_to_send,
                    caption="Your Upload",
                    width=200
                )
    
    st.divider()
    
    # ==========================================
    # Prediction Button & Results
    # ==========================================
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button(
            "🔍 Identify Character",
            type="primary",
            use_container_width=True
        )
    
    if predict_button:
        if image_to_send is None:
            st.warning("⚠️ Please draw something or upload an image first.")
        else:
            with st.spinner("🔄 Analyzing..."):
                try:
                    result = send_prediction_request(image_to_send)
                    
                    if result["success"]:
                        data = result["data"]
                        
                        # Display results
                        col_pred, col_conf = st.columns(2)
                        
                        with col_pred:
                            st.metric(
                                label="Prediction",
                                value=data['prediction'],
                                help="The character the model thinks you drew"
                            )
                        
                        with col_conf:
                            confidence_pct = data['confidence'] * 100
                            st.metric(
                                label="Confidence",
                                value=f"{confidence_pct:.1f}%",
                                help="How confident the model is in this prediction"
                            )
                        
                        # Confidence bar
                        st.progress(
                            data['confidence'],
                            text=f"Confidence: {confidence_pct:.1f}%"
                        )
                        
                        # Low confidence warning
                        if data['confidence'] < 0.5:
                            st.warning(
                                "⚠️ Low confidence. Try drawing the character "
                                "larger and more clearly."
                            )
                    else:
                        st.error(result["error"])
                        
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to the prediction API. "
                        "Please ensure the backend is running."
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # ==========================================
    # Footer
    # ==========================================
    st.divider()
    st.caption(
        "Built with ❤️ by David Osei Kumi | "
        "[GitHub](https://github.com/dkumi12) | "
        "[LinkedIn](https://www.linkedin.com/in/daniel-kumi-9b5834205/)"
    )


# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    main()
