import streamlit as st
import cv2
import numpy as np
import pickle

IMAGE_SIZE = 128

# Load trained model
with open("rail_track_crack_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Rail Track Crack Detection", layout="centered")

st.title("🚆 Rail Track Crack Detection System")
st.write("Upload a railway track image to detect cracks using Machine Learning.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_flattened = img_resized.flatten().reshape(1, -1)

        prediction = model.predict(img_flattened)[0]

        st.image(img, caption="Uploaded Image", width=400)

        if prediction == 1:
            st.error("🔴 Defective Track (Crack Detected)")
        else:
            st.success("🟢 Non Defective Track (No Crack Detected)")
