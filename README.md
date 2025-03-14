# Image Processing Application

A Streamlit-based application that provides various image processing capabilities including:
- Original image viewing with details
- Image blurring with adjustable kernel size
- Edge detection with adjustable thresholds
- Raccoon detection using YOLOv8 (using class ID 15)

## Installation

To install all required dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the Application

To start the application, run:
```bash
streamlit run app.py
```

The application will open in your default web browser. You can stop the application at any time by pressing `Ctrl+C` in the terminal.

## Features

- **Upload**: Supports JPG, JPEG, and PNG formats
- **Processing Options**:
  - Original: View uploaded image with details
  - Blur: Apply adjustable Gaussian blur
  - Edge Detection: Canny edge detection with adjustable thresholds
  - Raccoon Detection: AI-powered raccoon detection with confidence slider
- **Download**: Each processed image can be downloaded
- **Real-time Updates**: All adjustments are reflected immediately in the interface

## Dependencies

The application requires the following Python packages:
- streamlit==1.31.1
- opencv-python-headless==4.8.1.78
- numpy==1.26.3
- torch==2.1.2
- ultralytics==8.1.2
- Pillow==10.2.0 