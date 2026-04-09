import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import plotly.express as px
import pandas as pd

# 1. Page Config
st.set_page_config(page_title="AI Waste Audit", page_icon="♻️", layout="wide")

# 2. Custom Header
st.title("♻️ AI Ultimate Trash Classifier & Audit")
st.markdown("### *Use your camera or upload photos to get a complete percentage analysis!*")
st.divider()

@st.cache_resource
def load_model():
    # Loading your smart AI model
    return tf.keras.models.load_model("trash_model_v3.keras", compile=False)

try:
    model = load_model()
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
except Exception as e:
    st.error(f"Error loading model. Details: {e}")
    st.stop()

# 3. Create Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### 📸 Input Methods")
    
    # 1st Option: Live Camera
    camera_photo = st.camera_input("Take a picture of the waste")
    st.markdown("---")
    
    # 2nd Option: Batch File Uploader
    uploaded_files = st.file_uploader("Or upload images (batch allowed)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Combine all images (camera + uploaded) into one list to process
images_to_process = []
if camera_photo:
    images_to_process.append({"file": camera_photo, "name": "Camera Capture"})
if uploaded_files:
    for f in uploaded_files:
        images_to_process.append({"file": f, "name": f.name})

with col2:
    if images_to_process:
        st.markdown("#### 🤖 AI Audit Results")
        
        # Trackers for our data
        waste_counts = {category: 0 for category in classes}
        individual_results = []
        
        my_bar = st.progress(0, text="Scanning images...")
        
        # Process every image
        for i, item in enumerate(images_to_process):
            image = Image.open(item["file"]).convert('RGB')
            
            # --- FIX 1: CROP WITHOUT SQUISHING ---
            size = (224, 224)
            processed_image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            
            # --- FIX 2: THE PERFECT MATH ---
            img_array = np.array(processed_image, dtype=np.float32)
            img_array = (img_array / 127.5) - 1.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make Prediction
            predictions = model.predict(img_array, verbose=0)[0]
            top_idx = np.argmax(predictions)
            confidence = predictions[top_idx] * 100
            result = classes[top_idx]
            
            # Store Results
            waste_counts[result] += 1
            individual_results.append({
                "image": processed_image,
                "filename": item["name"],
                "category": result.upper(),
                "confidence": confidence,
                "raw_predictions": predictions
            })
            
            my_bar.progress((i + 1) / len(images_to_process), text=f"Scanned {i+1} of {len(images_to_process)} images...")
            
        my_bar.empty()

        # --- TOTAL PERCENTAGE ANALYSIS (CHART) ---
        df = pd.DataFrame(list(waste_counts.items()), columns=['Waste Type', 'Count'])
        df = df[df['Count'] > 0] # Hide categories with 0 items
        
        if not df.empty:
            fig = px.pie(df, values='Count', names='Waste Type', hole=0.4, 
                         title="Total Waste Composition Analysis",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
        # --- INDIVIDUAL BREAKDOWN ---
        st.divider()
        st.markdown("#### 🔍 Individual Item Breakdown")
        
        # Show results in a clean grid
        cols = st.columns(3)
        for idx, res in enumerate(individual_results):
            with cols[idx % 3]:
                # Show what the AI actually looked at (cropped perfect square)
                st.image(res["image"], use_column_width=True)
                st.success(f"**{res['category']}** ({res['confidence']:.1f}%)")
                st.caption(res["filename"])
                
                # The Debugger (See exactly what the AI is thinking)
                with st.expander("See Raw Percentages"):
                    for j, class_name in enumerate(classes):
                        st.write(f"{class_name.capitalize()}: {res['raw_predictions'][j]*100:.1f}%")

        # Celebration only if it successfully stops guessing Cardboard for everything!
        if any(res['category'] != 'CARDBOARD' for res in individual_results):
            st.balloons()
