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

# ---------------- Upload path ----------------
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload a handwritten expression image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  # Grayscale

# ---------------- Draw path ----------------
elif input_method == "Draw Expression":
    st.write("Draw your expression below:")

    # session counter to reset canvas when clearing
    if "canvas_counter" not in st.session_state:
        st.session_state.canvas_counter = 0

    # Tool selection and stroke width (defaults depend on tool)
    tool = st.radio("Tool:", ["Brush", "Eraser"], index=0, horizontal=True)
    default_width = 8 if tool == "Brush" else 30
    stroke_width = st.slider("Stroke width", min_value=1, max_value=60, value=default_width)

    # Clear button - increments counter to change canvas key (resets it)
    if st.button("Clear Canvas"):
        st.session_state.canvas_counter += 1

    # Single canvas whose behavior depends on 'tool'
    canvas_key = f"canvas_{st.session_state.canvas_counter}"
    stroke_color = "black" if tool == "Brush" else "white"   # white = erase on white background

    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="white",
        width=560,
        height=560,
        drawing_mode="freedraw",
        key=canvas_key,
    )

    # Convert canvas RGBA -> grayscale PIL Image
    if canvas_result.image_data is not None:
        rgba = canvas_result.image_data.astype(np.uint8)
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        image = Image.fromarray(gray)

# ---------------- Display & Submit ----------------
if image:
    st.image(image, caption="Input Expression", use_container_width=True)

    if st.button("Submit Expression"):
        symbols = preprocess_and_segment(image)

        if symbols:
            st.write("### Segmented Symbols")
            cols = st.columns(len(symbols))
            for i, sym in enumerate(symbols):
                cols[i].image(sym, width=50, caption=f"Symbol {i+1}")
        else:
            st.warning("No symbols detected. Try another image.")
