# Vehicle Detection with YOLOv12

This is a real-time vehicle detection application built with YOLOv12 and Gradio. It allows you to upload videos and visualize vehicle detections using a fine-tuned YOLOv8 model.

## Features

- Easy-to-use web interface with Gradio
- Upload and process any video file
- Real-time visualization of vehicle detection
- Processed videos can be downloaded

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/vehicle-detection.git
cd vehicle-detection
pip install -r requirements.txt
```

## Usage

Run the application with:

```bash
python app.py
```

Then open your web browser and navigate to the provided URL (typically http://127.0.0.1:7860).

1. Upload a video using the "Upload Video" section
2. Click the "Submit" button to process the video
3. View the results in the "Processed Video" section
4. Download the processed video if needed

## Model

The application uses a fine-tuned YOLOv8 model (`best.pt`) trained specifically for vehicle detection. The model can detect various types of vehicles with high accuracy.

## Deployment

This application can be easily deployed to Hugging Face Spaces:

1. Fork this repository
2. Create a new Space on Hugging Face
3. Link your GitHub repository to the Space
4. Select Gradio as the SDK
5. Deploy!

## License

MIT


