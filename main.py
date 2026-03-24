import streamlit as st
from teachable_machine import TeachableMachine
from PIL import Image
import numpy as np

MODEL_URL = "https://teachablemachine.withgoogle.com/models/4X9q3hZoX/"

# Initialize the model using the cloud link
# Note: This library will download the necessary weights from your link
model = TeachableMachine(model_path=MODEL_URL)

st.title("Cloud-Based Image Classifier")
st.write("Using Teachable Machine hosted model")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a PIL Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Save temporarily to classify (the library often expects a file path or array)
    image.save("temp_image.jpg")

    # Run classification
    # result returns a dictionary with class_name, class_index, and confidence
    result = model.classify_image("temp_image.jpg")

    st.divider()
    st.subheader("Result")
    st.write(f"**Class:** {result['class_name']}")
    st.write(f"**Confidence:** {result['class_confidence']:.2%}")
