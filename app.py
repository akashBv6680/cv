# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import cv2

# --- Model and Configuration ---
# Use the YOLOv8 Nano model (.pt extension) for fast inference on Streamlit Cloud (no GPU)
# This file must be available in your GitHub repository or downloaded during runtime.
MODEL_PATH = "yolov8n.pt" 

# Use a specific fine-tuned model path if you have one. 
# Example: FINE_TUNED_MODEL_PATH = "runs/detect/train/weights/best.pt"

# --- Main Functions ---

@st.cache_resource 
def load_model():
    """Loads the YOLOv8 model using st.cache_resource for efficiency."""
    # This function runs only once per deployment
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None

def detect_objects(model, image, conf_threshold):
    """Performs object detection and returns the annotated image."""
    # The 'stream=True' is critical for plotting the results correctly in Streamlit/OpenCV
    results = model.predict(
        source=image, 
        conf=conf_threshold, 
        save=False, 
        stream=True, 
        device='cpu' # Streamlit Cloud usually runs on CPU
    )
    
    # Process the results (we take the first result as input is a single image)
    for r in results:
        # Ultralytics results.plot() returns the annotated frame as a NumPy array (BGR format)
        annotated_frame = r.plot() 
        # Convert BGR (OpenCV default) to RGB for Streamlit/PIL display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        return annotated_frame_rgb
    
    return np.array(image) # Return original if no detections occur


# --- Streamlit App Layout ---

st.title("Object Detection with YOLOv8 & Streamlit")
st.write("Upload an image to detect objects using a pre-trained YOLOv8 model.")

# Load the model
yolo_model = load_model()

if yolo_model:
    # Sidebar for Model Configuration
    st.sidebar.header("Model Parameters")
    confidence = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.25
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload an Image (.jpg, .jpeg, .png)", 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Read the uploaded file into a PIL Image
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Display Original Image
        st.subheader("Original Image")
        st.image(image, caption='Image uploaded successfully.', use_column_width=True)

        # Detection Button
        if st.button("Run Object Detection"):
            with st.spinner('Detecting objects... This may take a moment on CPU.'):
                # Run the detection
                detected_image_array = detect_objects(yolo_model, image, confidence)
                
                # Display Detected Image
                st.subheader("Detected Objects")
                st.image(detected_image_array, caption='Detection Results', use_column_width=True)
                
                # Optional: Display detection summary
                st.success("Detection Complete!")
