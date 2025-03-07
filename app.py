import gradio as gr
import torch
import numpy as np
import cv2
import os
import tempfile
from pathlib import Path
from ultralytics import YOLO

# Load the YOLO model
model_path = Path(__file__).parent / "best.pt"
model = YOLO(model_path)

def process_video(video_path):
    """
    Process a video with the YOLO model and return the processed video path
    """
    if not video_path:
        return None
    
    # Create temporary file for output
    temp_output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # Process video with YOLO
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define codec and create VideoWriter object
    output = cv2.VideoWriter(
        temp_output_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, 
        (width, height)
    )
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Write the frame to the output video
        output.write(annotated_frame)
        
    # Release resources
    cap.release()
    output.release()
    
    return temp_output_path

# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Vehicle Detection with YOLOv12")
    gr.Markdown("Upload a video and click 'Submit' to detect vehicles using a fine-tuned YOLOv12 model.")
    
    with gr.Row():
        input_video = gr.Video(label="Upload Video")
        output_video = gr.Video(label="Processed Video")
    
    submit_btn = gr.Button("Submit")
    submit_btn.click(
        fn=process_video,
        inputs=[input_video],
        outputs=[output_video]
    )

if __name__ == "__main__":
    app.launch()