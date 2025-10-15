import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# --- Configuration ---
# Ensure yolov8n.pt is in the same directory as app.py
MODEL_PATH = 'yolov8n.pt'
st.set_page_config(
    page_title="YOLOv8 Image Detector",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Model Loading with Caching ---
# @st.cache_resource ensures the model is loaded only once across the entire app lifetime,
# which is crucial for performance and avoids the model loading error repeatedly.
@st.cache_resource
def load_yolo_model(path):
    """Loads the YOLO model and handles potential load errors."""
    try:
        # The YOLO class from ultralytics handles the necessary PyTorch
        # safe global serialization for the checkpoint file (yolov8n.pt).
        model = YOLO(path)
        return model
    except Exception as e:
        # Display a custom error if the model fails to load
        st.error(f"Failed to load the model from {path}.")
        st.error("This usually happens due to mismatched dependency versions (PyTorch/Ultralytics) or a missing model file.")
        st.error(f"Detailed Error: {e}")
        return None

# Load the model
model = load_yolo_model(MODEL_PATH)

# --- Application UI ---
st.title("Object Detection with YOLOv8 (Streamlit)")
st.caption("Upload an image to detect objects using the YOLOv8 Nano model.")

if model:
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image file (PNG, JPG, or JPEG) to run object detection."
    )

    if uploaded_file is not None:
        # 1. Read the image
        try:
            # Open the image using PIL
            image = Image.open(uploaded_file)
            st.image(image, caption='Original Image', use_column_width=True)
            st.write("Running detection...")
            
            # 2. Run inference
            # The 'source' parameter takes a PIL Image object directly
            results = model.predict(
                source=image, 
                conf=0.25, # Confidence threshold
                iou=0.7,   # IOU threshold for Non-Maximum Suppression
                save=False
            )

            # 3. Get the annotated image
            # The .plot() method returns a NumPy array with bounding boxes and labels drawn.
            annotated_image = results[0].plot()

            # 4. Display the results
            st.subheader("Detection Results")
            st.image(annotated_image, caption='Detected Objects', use_column_width=True)

            # Optional: Display detection summary
            detected_classes = results[0].boxes.cls.tolist()
            if detected_classes:
                class_names = results[0].names
                class_counts = {}
                for cls_index in detected_classes:
                    name = class_names[int(cls_index)]
                    class_counts[name] = class_counts.get(name, 0) + 1
                
                st.markdown("---")
                st.subheader("Summary of Detections")
                for name, count in class_counts.items():
                    st.write(f"- **{name.capitalize()}**: {count} instance(s)")
            else:
                st.info("No objects detected in the image with the current confidence threshold.")

        except Exception as e:
            st.error(f"An error occurred during detection: {e}")
            st.error("Please ensure your uploaded file is a valid image.")

# Optional placeholder for initial state
else:
    st.info("Waiting for model to load and initialization to complete. Please ensure all dependency files are correct.")
