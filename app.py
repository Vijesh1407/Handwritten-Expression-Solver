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
        background-color: #211C1C;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 1.5em;
        text-align: center;
    }
    .result-box {
        background-color: #211C1C;
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

# Initialize session state for manual corrections
if 'predicted_symbols' not in st.session_state:
    st.session_state.predicted_symbols = []
if 'manual_corrections' not in st.session_state:
    st.session_state.manual_corrections = {}
if 'symbols_data' not in st.session_state:
    st.session_state.symbols_data = []
if "canvas_counter" not in st.session_state:
    st.session_state.canvas_counter = 0

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

# ---------------- Draw Expression ----------------
st.subheader("🎨 Draw Your Expression")

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
        st.session_state.predicted_symbols = []
        st.session_state.manual_corrections = {}
        st.session_state.symbols_data = []
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

image = None
if canvas_result.image_data is not None:
    # Check if canvas has content
    if np.sum(canvas_result.image_data[:, :, 3]) > 0:  # Check alpha channel
        rgba = canvas_result.image_data.astype(np.uint8)
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        image = Image.fromarray(gray)

# ---------------- Process Expression ----------------
st.markdown("---")

if image:
    if st.button("🚀 Analyze Expression", type="primary", use_container_width=True):
        with st.spinner("🔍 Analyzing your handwriting..."):
            try:
                # Reset corrections
                st.session_state.manual_corrections = {}
                
                # Segment the image
                symbols = preprocess_and_segment(image)
                
                if not symbols:
                    st.markdown('<div class="error-box">⚠️ No symbols detected.<br><small>Try writing larger with better spacing</small></div>', unsafe_allow_html=True)
                    st.stop()
                
                # Predict each symbol and store data
                st.session_state.symbols_data = []
                st.session_state.predicted_symbols = []
                
                for i, sym in enumerate(symbols):
                    # Normalize
                    sym_norm = sym.astype("float32") / 255.0
                    sym_norm = np.expand_dims(sym_norm, axis=(0, -1))
                    
                    # Predict with the model
                    predictions = model.predict(sym_norm, verbose=0)[0]
                    
                    # Get top 5 predictions
                    top_5_idx = np.argsort(predictions)[-5:][::-1]
                    
                    # Store symbol data
                    st.session_state.symbols_data.append({
                        'image': sym,
                        'predictions': predictions,
                        'top_5_idx': top_5_idx,
                        'top_prediction': label_map[top_5_idx[0]],
                        'confidence': predictions[top_5_idx[0]]
                    })
                    
                    st.session_state.predicted_symbols.append(label_map[top_5_idx[0]])
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
                st.info("Please try again with a clearer drawing.")

# Display symbols with manual correction options
if st.session_state.symbols_data:
    st.markdown("### 🧩 Detected Symbols (Select correct symbol from dropdown)")
    st.write(f"Found **{len(st.session_state.symbols_data)}** symbols")
    
    # Create columns for symbols
    num_symbols = len(st.session_state.symbols_data)
    cols_per_row = 6
    
    for row_start in range(0, num_symbols, cols_per_row):
        cols = st.columns(min(cols_per_row, num_symbols - row_start))
        
        for col_idx, i in enumerate(range(row_start, min(row_start + cols_per_row, num_symbols))):
            sym_data = st.session_state.symbols_data[i]
            
            with cols[col_idx]:
                # Show the symbol image
                st.image(sym_data['image'], width=70)
                
                # Get top 5 options with labels
                top_5_options = []
                for idx in sym_data['top_5_idx']:
                    label = label_map[idx]
                    conf = sym_data['predictions'][idx]
                    top_5_options.append(f"{label} ({conf:.1%})")
                
                # Get current selection
                if i in st.session_state.manual_corrections:
                    current_label = st.session_state.manual_corrections[i]
                else:
                    current_label = sym_data['top_prediction']
                
                # Find current index in options
                try:
                    current_idx = [opt.split(' ')[0] for opt in top_5_options].index(current_label)
                except ValueError:
                    current_idx = 0
                
                # Confidence color indicator
                confidence = sym_data['confidence']
                if confidence > 0.85:
                    conf_color = "🟢"
                elif confidence > 0.6:
                    conf_color = "🟡"
                else:
                    conf_color = "🔴"
                
                # Dropdown for manual selection
                selected = st.selectbox(
                    f"#{i+1} {conf_color}",
                    options=top_5_options,
                    index=current_idx,
                    key=f"symbol_select_{i}",
                    label_visibility="visible"
                )
                
                # Extract selected label
                selected_label = selected.split(' ')[0]
                
                # Update manual correction if changed
                if selected_label != sym_data['top_prediction']:
                    st.session_state.manual_corrections[i] = selected_label
                elif i in st.session_state.manual_corrections:
                    del st.session_state.manual_corrections[i]
    
    st.markdown("---")
    
    # Get final labels (with manual corrections applied)
    final_labels = []
    for i, sym_data in enumerate(st.session_state.symbols_data):
        if i in st.session_state.manual_corrections:
            final_labels.append(st.session_state.manual_corrections[i])
        else:
            final_labels.append(sym_data['top_prediction'])
    
    # Format and display expression
    formatted_expr = format_expression(final_labels)
    st.markdown(f'<div class="expression-box">📝 Expression: <strong>{formatted_expr}</strong></div>', unsafe_allow_html=True)
    
    # Calculate confidence metrics
    confidence_scores = [sym['confidence'] for sym in st.session_state.symbols_data]
    avg_confidence = np.mean(confidence_scores)
    min_confidence = np.min(confidence_scores)
    
    # Show confidence metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Confidence", f"{avg_confidence:.1%}")
    with col2:
        st.metric("Lowest Confidence", f"{min_confidence:.1%}")
    with col3:
        st.metric("Corrections Made", len(st.session_state.manual_corrections))
    
    # Warning for low confidence
    if min_confidence < 0.5 and not st.session_state.manual_corrections:
        st.warning(f"⚠️ Very low confidence. Please review and correct symbols above!")
    elif avg_confidence < 0.7 and not st.session_state.manual_corrections:
        st.info("ℹ️ Moderate confidence. Please review the predictions above.")
    
    # Parse and evaluate
    st.markdown("### 🧮 Result")
    expr, result = parse_and_evaluate(final_labels)
    
    if isinstance(result, (int, float)):
        st.markdown(f'<div class="result-box">✅ <strong>{result}</strong></div>', unsafe_allow_html=True)
        st.balloons()
    elif isinstance(result, str) and "✓" in result:
        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f'<div class="error-box">⚠️ {result}</div>', unsafe_allow_html=True)
        st.info("💡 Tip: Check if all symbols are correct above and make corrections if needed.")

else:
    st.info("👆 Please draw an expression above to get started.")
    
    # Show example
    with st.expander("📖 See Examples & Tips"):
        st.markdown("""
        ### ✨ Feature: Manual Correction!
        After analyzing your expression, you can:
        - Review each detected symbol
        - See top 5 predictions with confidence scores
        - **Manually select the correct symbol** from dropdown
        - Correct any misclassified symbols before calculation
        
        ### Good Examples:
        - `2+3=5` → Should recognize as 2, +, 3, =, 5
        - `15-7=8` → Should recognize as 1, 5, -, 7, =, 8
        - `3*4=12` → Should recognize as 3, *, 4, =, 1, 2
        
        ### Common Issues:
        - **Too small**: Write larger, use more canvas space
        - **Too close**: Leave gaps between symbols
        - **Unclear**: Make symbols distinct (especially + and =)
        - **Wrong prediction**: Use the dropdown to correct it!
        """)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Supported Symbols:</strong> 0-9, +, -, ×, ÷, =</p>
        <p>📝 <strong>Pro Tip:</strong> Write LARGE with WIDE spacing between symbols!</p>
        <p>⚠️ <strong>Equals Sign:</strong> Draw TWO separate horizontal lines, not one thick line</p>
        <p>✨ <strong>New:</strong> Manually correct any misclassified symbols using the dropdowns!</p>
    </div>
""", unsafe_allow_html=True)



