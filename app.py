import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="Trash Classifier", page_icon="♻️", layout="wide")

# 2. Custom Header
st.title("♻️ AI Live Trash Classifier")
st.markdown("### *Hold your waste up to the camera or upload a photo!*")
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

# 3. Create Two Columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📸 Input Method")
    
    # Add the live camera widget!
    camera_photo = st.camera_input("Take a picture of the waste")
    
    st.markdown("---") # A subtle line separator
    
    # Keep the uploader as a backup option
    uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])
    
    # Figure out which image the user provided
    image_to_process = None
    if camera_photo is not None:
        image_to_process = camera_photo
    elif uploaded_file is not None:
        image_to_process = uploaded_file

with col2:
    if image_to_process is not None:
        st.markdown("#### 🤖 AI Analysis")
        
        # Open the image (works for both camera and uploaded files!)
        image = Image.open(image_to_process).convert('RGB')
        
        with st.spinner("Analyzing the materials..."):
            # Prepare the image for the AI
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make the prediction
            predictions = model.predict(img_array, verbose=0)[0]
            
            top_idx = np.argmax(predictions)
            confidence = predictions[top_idx] * 100
            result = classes[top_idx].upper()
            
        # Display the results
        st.success(f"**FINAL DECISION:** Put it in the {result} bin!")
        st.info(f"**AI CONFIDENCE:** {confidence:.2f}%")
        
        st.balloons()
