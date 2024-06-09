# main_app.py

import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Function to preprocess uploaded image
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Function to make prediction
def predict(img):
    model = load_model('ACL_PREDICTION.keras')  # Load your trained model
    
    # Resize the input image to match the expected input shape of the model
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values

    prediction = model.predict(img_array)[0][0]  # Assuming binary classification output
    return prediction


def main():
    st.title('ACL Injury Prediction')

    # Upload image from file
    uploaded_file = st.file_uploader("Upload Image (JPEG, PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
    else:
        st.warning("Please upload an image.")
        return

    st.image(img, caption='Uploaded Image', use_column_width=True)
    prediction = predict(img)
    inverted_prediction = 1 - prediction
    if inverted_prediction < 0.5:
        st.write("Predicted Label: No Injury")
    else:
        st.write("Predicted : Injury")
    st.write(f'Predicted risk of injury: {inverted_prediction*100:.1f}%')

if __name__ == '__main__':
    main()
