import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Trash Classifier", page_icon="♻️")
st.title("♻️ AI Trash & Recycling Classifier")
st.write("Upload an image of waste, and the AI will tell you how to dispose of it!")

# 2. Load the Model (Cached for speed)
@st.cache_resource
def load_model():
    # Make sure this perfectly matches the name of your model file!
    return tf.keras.models.load_model("trash_model_v3.keras")

try:
    model = load_model()
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
except Exception as e:
    st.error(f"Error loading model. Check if your .keras file is in the folder. Details: {e}")
    st.stop()

# 3. Web UI: Image Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 4. Processing and Prediction
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Analyzing...")

    # Preprocess the image exactly like your local script did
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]

    # Get results
    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx] * 100
    result = classes[top_idx].upper()

    # Display Results beautifully using Streamlit
    st.success(f"**FINAL DECISION:** {result}")
    st.info(f"**CONFIDENCE:** {confidence:.2f}%")

    # Optional: Detailed Breakdown expander
    with st.expander("See detailed breakdown"):
        for cls, score in zip(classes, predictions):
            st.write(f"- **{cls.capitalize()}**: {score*100:.1f}%")