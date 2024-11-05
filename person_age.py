import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained age prediction model
model = tf.keras.models.load_model("age_prediction_model3.h5")

# Define image preprocessing function
def preprocess_image(image):
    # Resize and normalize image
    image = image.resize((200, 200), Image.LANCZOS)
    image_array = np.asarray(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app
st.title("Age Prediction App")
st.write("Upload an image to predict the age of the person.")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    preprocessed_image = preprocess_image(image)

    # Make prediction
    age_prediction = model.predict(preprocessed_image)
    age = age_prediction[0][0]  # Extract the predicted age

    # Display the predicted age
    st.write(f"Predicted Age: {int(age)} years")
