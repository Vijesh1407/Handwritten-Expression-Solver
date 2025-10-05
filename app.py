import streamlit as st
import numpy as np
from PIL import Image
import cv2
import json
import os
from preprocessing.segment import preprocess_and_segment
from utils.parser import parse_and_evaluate, format_expression
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Handwritten Expression Solver",
    page_icon="✍️",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .expression-box {
        background-color: #292222;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 1.5em;
        text-align: center;
    }
    .result-box {
        background-color: #292222;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 1.3em;
        text-align: center;
        border: 2px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        border: 2px solid #dc3545;
    }
    .symbol-card {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        background: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">✍️ Handwritten Expression Solver</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- Load model & class indices ----------------
@st.cache_resource
def load_model_and_classes():
    """Load model and class indices with caching"""
    try:
        if not os.path.exists("models/math_symbol_cnn.keras"):
            st.error("❌ Model file not found. Please run train_model.py first!")
            return None, None
        
        if not os.path.exists("models/class_indices.json"):
            st.error("❌ Class indices file not found. Please run train_model.py first!")
            return None, None
        
        model = load_model("models/math_symbol_cnn.keras")
        
        with open("models/class_indices.json") as f:
            class_indices = json.load(f)
        
        label_map = {v: k for k, v in class_indices.items()}
        
        return model, label_map
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

model, label_map = load_model_and_classes()

if model is None or label_map is None:
    st.stop()

# Display model info
with st.expander("ℹ️ Model Information"):
    st.write(f"**Classes supported:** {len(label_map)}")
    st.write(f"**Symbols:** {', '.join(sorted(label_map.values()))}")
    
    # Load training history if available
    if os.path.exists("models/training_history.json"):
        with open("models/training_history.json") as f:
            history = json.load(f)
        if 'best_val_accuracy' in history:
            st.write(f"**Model Accuracy:** {history['best_val_accuracy']*100:.2f}%")

# ---------------- Input method selection ----------------
st.subheader("📝 Choose Input Method")
input_method = st.radio(
    "How would you like to input your expression?",
    ["📤 Upload Image", "🎨 Draw Expression"],
    horizontal=True
)

image = None

# ---------------- Upload Image ----------------
if input_method == "📤 Upload Image":
    st.markdown("### Upload Your Handwritten Expression")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear image of your handwritten mathematical expression"
        )
    
    with col2:
        st.info("**Tips:**\n- Clear writing\n- Good spacing\n- Dark on light")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", use_column_width=True)

# ---------------- Draw Expression ----------------
elif input_method == "🎨 Draw Expression":
    st.markdown("### Draw Your Expression")
    
    # Canvas state management
    if "canvas_counter" not in st.session_state:
        st.session_state.canvas_counter = 0
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        tool = st.selectbox("🛠️ Tool:", ["Brush", "Eraser"])
    
    with col2:
        default_width = 12 if tool == "Brush" else 30
        stroke_width = st.slider("📏 Stroke Width:", 5, 50, default_width)
    
    with col3:
        st.write("")  # Spacing
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.session_state.canvas_counter += 1
            st.rerun()
    
    # Drawing tips
    with st.expander("💡 Drawing Tips for Best Accuracy"):
        st.markdown("""
        - **Write BIG**: Use 70-80% of the canvas height
        - **Spacing**: Leave LARGE gaps between symbols (at least 1.5 symbol widths)
        - **Stroke width**: Use 12-15 for best results
        - **Line quality**: Write smoothly, avoid shaky lines
        - **Write slowly and deliberately** - rushed writing causes errors
        
        **Critical Symbol Tips:**
        - **Numbers (0-9)**: Write clearly, make each digit distinct
          - `2`: Curved top, straight diagonal, flat bottom
          - `4`: Clear closed top, vertical line
          - `7`: Add a crossbar to distinguish from 1
        - **Plus (+)**: Two equal lines crossing in the center
        - **Equals (=)**: TWO horizontal parallel lines with clear spacing
          - Write the lines separately
          - Keep them the same length
          - Make sure they're aligned horizontally
        - **Minus (-)**: Single horizontal line
        - **Multiply (×)**: X shape with equal arms
        - **Divide (÷)**: Horizontal line with dots above and below
        
        **Common Mistakes to Avoid:**
        - Don't let symbols touch each other
        - Don't write too small or too large
        - Don't rush - take your time
        - Don't write equals sign as one thick line - use TWO separate lines
        """)
    
    canvas_key = f"canvas_{st.session_state.canvas_counter}"
    stroke_color = "black" if tool == "Brush" else "white"
    
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="white",
        width=700,
        height=200,
        drawing_mode="freedraw",
        key=canvas_key,
        display_toolbar=False
    )
    
    if canvas_result.image_data is not None:
        # Check if canvas has content
        if np.sum(canvas_result.image_data[:, :, 3]) > 0:  # Check alpha channel
            rgba = canvas_result.image_data.astype(np.uint8)
            gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
            image = Image.fromarray(gray)

# ---------------- Process Expression ----------------
st.markdown("---")

if image:
    if st.button("🚀 Solve Expression", type="primary", use_container_width=True):
        with st.spinner("🔍 Analyzing your handwriting..."):
            try:
                # Segment the image
                symbols = preprocess_and_segment(image)
                
                if not symbols:
                    st.markdown('<div class="error-box">⚠️ No symbols detected.<br><small>Try writing larger with better spacing</small></div>', unsafe_allow_html=True)
                    st.stop()
                
                # Display segmented symbols
                st.markdown("### 🧩 Detected Symbols")
                st.write(f"Found **{len(symbols)}** symbols")
                
                # Predict each symbol
                predicted_labels = []
                confidence_scores = []
                
                # Create columns for symbols (max 12 per row)
                num_cols = min(len(symbols), 12)
                cols = st.columns(num_cols)
                
                for i, sym in enumerate(symbols):
                    # Normalize
                    sym_norm = sym.astype("float32") / 255.0
                    sym_norm = np.expand_dims(sym_norm, axis=(0, -1))
                    
                    # Predict with the model
                    predictions = model.predict(sym_norm, verbose=0)[0]
                    
                    # Get top 3 predictions
                    top_3_idx = np.argsort(predictions)[-3:][::-1]
                    pred_idx = top_3_idx[0]
                    confidence = predictions[pred_idx]
                    
                    pred_char = label_map[pred_idx]
                    
                    predicted_labels.append(pred_char)
                    confidence_scores.append(confidence)
                    
                    # Display in column
                    col_idx = i % num_cols
                    with cols[col_idx]:
                        # Show the symbol image
                        st.image(sym, width=70)
                        
                        # Show prediction with confidence
                        if confidence > 0.85:
                            conf_color = "🟢"
                        elif confidence > 0.6:
                            conf_color = "🟡"
                        else:
                            conf_color = "🔴"
                        
                        st.markdown(f"**{pred_char}** {conf_color}")
                        st.caption(f"{confidence:.1%}")
                        
                        # Show top 3 predictions in expander
                        with st.expander("Top 3"):
                            for idx in top_3_idx:
                                st.write(f"{label_map[idx]}: {predictions[idx]:.1%}")
                
                st.markdown("---")
                
                # Format and display expression
                formatted_expr = format_expression(predicted_labels)
                st.markdown(f'<div class="expression-box">📝 Expression: <strong>{formatted_expr}</strong></div>', unsafe_allow_html=True)
                
                # Calculate average confidence
                avg_confidence = np.mean(confidence_scores)
                min_confidence = np.min(confidence_scores)
                
                # Show confidence metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")
                with col2:
                    st.metric("Lowest Confidence", f"{min_confidence:.1%}")
                
                # Warning for low confidence
                if min_confidence < 0.5:
                    st.warning(f"⚠️ Very low confidence on some symbols. Result may be incorrect!")
                elif avg_confidence < 0.7:
                    st.info("ℹ️ Moderate confidence. Double-check the result.")
                
                # Parse and evaluate
                st.markdown("### 🧮 Result")
                expr, result = parse_and_evaluate(predicted_labels)
                
                if isinstance(result, (int, float)):
                    st.markdown(f'<div class="result-box">✅ <strong>{expr} = {result}</strong></div>', unsafe_allow_html=True)
                    st.balloons()
                elif isinstance(result, str) and "✓" in result:
                    st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f'<div class="error-box">⚠️ {result}</div>', unsafe_allow_html=True)
                    st.info("💡 Tip: Check if all symbols were recognized correctly above.")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
                st.info("Please try again with a clearer image or drawing.")

else:
    st.info("👆 Please upload an image or draw an expression above to get started.")
    
    # Show example
    with st.expander("📖 See Examples"):
        st.markdown("""
        ### Good Examples:
        - `2+3=` → Should recognize as 2, +, 3, =
        - `15-7` → Should recognize as 1, 5, -, 7
        - `3*4=12` → Should recognize as 3, *, 4, =, 1, 2
        
        ### Common Issues:
        - **Too small**: Write larger, use more canvas space
        - **Too close**: Leave gaps between symbols
        - **Unclear**: Make symbols distinct (especially + and =)
        - **Sloppy**: Write neatly, avoid shaky lines
        """)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Supported Symbols:</strong> 0-9, +, -, ×, ÷, =</p>
        <p>📝 <strong>Pro Tip:</strong> Write LARGE with WIDE spacing between symbols!</p>
        <p>⚠️ <strong>Equals Sign:</strong> Draw TWO separate horizontal lines, not one thick line</p>
    </div>
""", unsafe_allow_html=True)

