import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import plotly.express as px
import pandas as pd
import cv2 # <--- The new computer vision library!

# 1. Page Config
st.set_page_config(page_title="AI Waste Audit", page_icon="♻️", layout="wide")

# 2. Custom Header
st.title("♻️ AI Ultimate Trash Classifier")
st.markdown("### *Use your camera, upload photos, or stream live video!*")
st.divider()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trash_model_v3.keras", compile=False)

try:
    model = load_model()
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
except Exception as e:
    st.error(f"Error loading model. Details: {e}")
    st.stop()

# --- TABS FOR DIFFERENT MODES ---
tab1, tab2 = st.tabs(["📸 Static Photos & Audit", "🎥 Live Video Scanner"])

# ==========================================
# TAB 1: THE EXISTING PHOTO & BATCH AUDIT
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 📸 Input Methods")
        camera_photo = st.camera_input("Take a picture of the waste")
        st.markdown("---")
        uploaded_files = st.file_uploader("Or upload images (batch allowed)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    images_to_process = []
    if camera_photo:
        images_to_process.append({"file": camera_photo, "name": "Camera Capture"})
    if uploaded_files:
        for f in uploaded_files:
            images_to_process.append({"file": f, "name": f.name})

    with col2:
        if images_to_process:
            st.markdown("#### 🤖 AI Audit Results")
            waste_counts = {category: 0 for category in classes}
            individual_results = []
            my_bar = st.progress(0, text="Scanning images...")
            
            for i, item in enumerate(images_to_process):
                image = Image.open(item["file"]).convert('RGB')
                
                size = (224, 224)
                processed_image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                
                img_array = np.array(processed_image, dtype=np.float32)
                img_array = (img_array / 127.5) - 1.0
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array, verbose=0)[0]
                top_idx = np.argmax(predictions)
                confidence = predictions[top_idx] * 100
                result = classes[top_idx]
                
                waste_counts[result] += 1
                individual_results.append({
                    "image": processed_image,
                    "filename": item["name"],
                    "category": result.upper(),
                    "confidence": confidence,
                    "raw_predictions": predictions
                })
                my_bar.progress((i + 1) / len(images_to_process))
                
            my_bar.empty()

            df = pd.DataFrame(list(waste_counts.items()), columns=['Waste Type', 'Count'])
            df = df[df['Count'] > 0] 
            
            if not df.empty:
                fig = px.pie(df, values='Count', names='Waste Type', hole=0.4, 
                             title="Total Batch Composition",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
            st.divider()
            cols = st.columns(3)
            for idx, res in enumerate(individual_results):
                with cols[idx % 3]:
                    st.image(res["image"], use_column_width=True)
                    st.success(f"**{res['category']}** ({res['confidence']:.1f}%)")
                    with st.expander("📊 Multi-Waste Breakdown"):
                        photo_data = pd.DataFrame({'Material': [c.capitalize() for c in classes], 'Percentage': res['raw_predictions'] * 100})
                        photo_data = photo_data[photo_data['Percentage'] > 1.0]
                        fig_mini = px.pie(photo_data, values='Percentage', names='Material', hole=0.3)
                        fig_mini.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, height=200)
                        fig_mini.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_mini, use_container_width=True)


# ==========================================
# TAB 2: LIVE REAL-TIME VIDEO FEED
# ==========================================
with tab2:
    st.markdown("#### 🎥 Real-Time Video Analysis")
    st.write("Check the box below to turn on your webcam. The AI will analyze the live feed continuously! *(Note: This works best when running the app locally)*")
    
    run_video = st.checkbox("🔴 Start Live Scanner")
    FRAME_WINDOW = st.image([]) # This acts as our video screen
    
    if run_video:
        # 0 opens the default laptop webcam
        camera = cv2.VideoCapture(0)
        
        while run_video:
            ret, frame = camera.read()
            if not ret:
                st.error("Could not access webcam.")
                break
                
            # OpenCV reads colors in BGR, we need RGB for Streamlit and AI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image to use our existing cropping math
            pil_img = Image.fromarray(frame_rgb)
            processed_image = ImageOps.fit(pil_img, (224, 224), Image.Resampling.LANCZOS)
            
            # The perfect math for MobileNet
            img_array = np.array(processed_image, dtype=np.float32)
            img_array = (img_array / 127.5) - 1.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict the current frame
            predictions = model.predict(img_array, verbose=0)[0]
            top_idx = np.argmax(predictions)
            confidence = predictions[top_idx] * 100
            result = classes[top_idx].upper()
            
            # --- Draw the result directly onto the live video ---
            cv2.putText(frame_rgb, f"{result} ({confidence:.1f}%)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Update the screen
            FRAME_WINDOW.image(frame_rgb)
    else:
        st.info("Scanner is paused.")
