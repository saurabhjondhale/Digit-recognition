import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2

# Load model
model = tf.keras.models.load_model("digit_model.h5")

st.title("📷🖼️ Handwritten Digit Recognizer")
st.write("Upload a photo or capture using camera")

option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

def preprocess_image(image: Image.Image):
    image = image.convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # White digit on black bg
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a digit image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Digit", width=150)
        img_array = preprocess_image(image)
        pred = model.predict(img_array)
        st.success(f"Predicted Digit: {np.argmax(pred)}")

elif option == "Use Camera":
    picture = st.camera_input("Take a photo of a digit")
    if picture is not None:
        image = Image.open(picture)
        st.image(image, caption="Captured Digit", width=150)
        img_array = preprocess_image(image)
        pred = model.predict(img_array)
        st.success(f"Predicted Digit: {np.argmax(pred)}")
