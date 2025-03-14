import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Simple Image App",
    page_icon="🦝",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .success-text {
        color: #28a745;
    }
    .info-text {
        color: #17a2b8;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model():
    """Cache the model loading to avoid reloading on every rerun"""
    model_path = 'yolov8s.pt'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    return YOLO(model_path)

def perform_blur(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def detect_edges(image, t1, t2):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img_gray, threshold1=t1, threshold2=t2)

def detect_raccoons(image, model, conf=0.6):
    results = model.predict(source=image, conf=conf)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detected_raccoons = 0
    for box in results[0].boxes:
        # Only process if class ID is 15 (cat -> raccoon)
        if int(box.cls[0]) == 15:  # Check class ID
            x1, y1, x2, y2 = box.xyxy[0]
            conf_score = float(box.conf[0])
            img_rgb = cv2.rectangle(img_rgb, 
                                  (int(x1), int(y1)), 
                                  (int(x2), int(y2)), 
                                  (255, 0, 0), 2)
            label = f"Raccoon: {conf_score:.2f}"
            # Add white background to text for better visibility
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_rgb, 
                         (int(x1), int(y1-label_height-10)), 
                         (int(x1+label_width), int(y1)), 
                         (255, 255, 255), 
                         -1)
            cv2.putText(img_rgb, label, (int(x1), int(y1-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            detected_raccoons += 1
    return img_rgb, detected_raccoons

def get_image_download_button(img, filename, text):
    """Generate a download button for the processed image"""
    buf = BytesIO()
    if len(img.shape) == 2:  # If image is grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:  # If image is BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(img).save(buf, format="PNG")
    return st.download_button(
        label=text,
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )

def main():
    # Main title with emoji
    st.title("Simple Image Processing App")
    
    # Two-column layout for better organization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📁 Upload Image")
        st.markdown("**Supported formats:** JPG, JPEG, PNG")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if not uploaded_file:
            st.warning("Please upload an image to begin processing")

    # Load YOLO model with progress indicator
    with st.spinner("🔄 Loading AI model..."):
        model = load_yolo_model()
        if model is None:
            return

    if uploaded_file is not None:
        try:
            # Process uploaded image
            with st.spinner("🔄 Processing uploaded image..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Display original image and process based on selection in second column
                with col2:
                    # Sidebar for processing options
                    with st.sidebar:
                        st.markdown("### 🛠️ Processing Options")
                        process_type = st.radio(
                            "Select Processing Type",
                            ["Original", "Blur", "Edge Detection", "Raccoon Detection"],
                            help="Choose the type of image processing to apply"
                        )
                        
                        # Show relevant parameters based on selection
                        if process_type == "Blur":
                            st.markdown("#### Blur Settings")
                            kernel_size = st.slider(
                                "Blur Strength",
                                3, 21, 5, step=2,
                                help="Higher values create stronger blur effect"
                            )
                        elif process_type == "Edge Detection":
                            st.markdown("#### Edge Detection Settings")
                            t1 = st.slider(
                                "Lower Threshold",
                                0, 500, 100,
                                help="Lower threshold for edge detection"
                            )
                            t2 = st.slider(
                                "Upper Threshold",
                                0, 500, 200,
                                help="Upper threshold for edge detection"
                            )
                        elif process_type == "Raccoon Detection":
                            st.markdown("#### Detection Settings")
                            confidence = st.slider(
                                "Confidence Threshold",
                                0.0, 1.0, 0.6,
                                help="Minimum confidence score for detection"
                            )

                    # Process and display based on selection
                    if process_type == "Original":
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        get_image_download_button(image, "original_image.png", "⬇️ Download Original Image")
                        
                        # Image info
                        height, width = image.shape[:2]
                        st.markdown(f"**Image Details:**")
                        st.markdown(f"- Dimensions: {width}x{height} pixels")
                        st.markdown(f"- File type: {uploaded_file.type}")
                        st.markdown(f"- File size: {uploaded_file.size/1024:.1f} KB")

                    elif process_type == "Blur":
                        blurred = perform_blur(image, kernel_size)
                        st.image(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
                        get_image_download_button(blurred, "blurred_image.png", "⬇️ Download Blurred Image")

                    elif process_type == "Edge Detection":
                        edges = detect_edges(image, t1, t2)
                        st.image(edges)
                        get_image_download_button(edges, "edge_detection.png", "⬇️ Download Edge Detection Result")

                    elif process_type == "Raccoon Detection":
                        result, num_raccoons = detect_raccoons(image, model, confidence)
                        st.image(result)
                        if num_raccoons > 0:
                            st.success(f"🦝 Detected {num_raccoons} raccoon(s) in the image!")
                        else:
                            st.info("No raccoons detected in this image")
                        get_image_download_button(result, "raccoon_detection.png", "⬇️ Download Detection Result")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.markdown("Please try uploading a different image or adjusting the parameters.")

if __name__ == "__main__":
    main() 