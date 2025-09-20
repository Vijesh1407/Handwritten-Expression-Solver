import streamlit as st
import numpy as np
from PIL import Image
import cv2
from preprocessing.segment import preprocess_and_segment
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Handwritten Expression Solver", layout="centered")
st.title("✍️ Handwritten Expression Solver")

# --- Option: Upload or Draw ---
st.write("Choose Input Method:")
input_method = st.radio("Input method:", ["Upload Image", "Draw Expression"])

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload a handwritten expression image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  # Grayscale

elif input_method == "Draw Expression":
    canvas_result = st_canvas(
        fill_color="black",       # Color for drawing
        stroke_width=8,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        # Convert the drawn canvas (numpy array) to PIL image
        drawn_image = canvas_result.image_data.astype(np.uint8)
        image = Image.fromarray(cv2.cvtColor(drawn_image, cv2.COLOR_RGBA2GRAY))

# --- Display Input Image ---
if image:
    st.image(image, caption="Input Expression", use_container_width=True)

    # --- Preprocess and Segment ---
    symbols = preprocess_and_segment(image)

    if symbols:
        st.write("### Segmented Symbols")
        cols = st.columns(len(symbols))
        for i, sym in enumerate(symbols):
            cols[i].image(sym, width=50, caption=f"Symbol {i+1}")
    else:
        st.warning("No symbols detected. Try another image.")
