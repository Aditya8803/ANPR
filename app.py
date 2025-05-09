import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import re
from google.cloud import vision

# ✅ Set up authentication using a service account JSON file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\ADITYA PARASHAR\Downloads\service_account.json"

# ✅ Initialize Google Vision API Client
client = vision.ImageAnnotatorClient()

# ✅ Function to extract license plate text using Google Cloud Vision API
def extract_license_plate(image):
    try:
        _, encoded_image = cv2.imencode('.jpg', image)
        image_bytes = encoded_image.tobytes()

        vision_image = vision.Image(content=image_bytes)
        response = client.text_detection(image=vision_image)
        texts = response.text_annotations

        if texts:
            return texts[0].description.strip()
        return None
    except Exception as e:
        return None


# ✅ Function to validate license plate format
def validate_license_plate(plate):
    pattern = r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$"  # Format: AA00AA0000
    return re.findall(pattern, plate)


# ✅ Streamlit UI
st.title("License Plate Recognition App")
mode = st.radio("Select Mode", ["Image", "Video"], index=0)

if mode == "Image":
    st.write("Upload an image to detect and extract the license plate number.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        license_plate_text = extract_license_plate(image)

        if license_plate_text and license_plate_text:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write(f"**Detected License Plate:** `{license_plate_text}`")
        else:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.error("No valid license plate detected in the image.")

else:
    st.write("Upload a video to detect and extract license plates in real time.")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            license_plate_text = extract_license_plate(frame)

            if license_plate_text:
                cv2.putText(frame, f"Plate: {license_plate_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        os.remove(video_path)
        st.success("Processing completed!")
