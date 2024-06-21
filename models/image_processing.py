# app/image_processing.py
import torch
import numpy as np
from PIL import Image

class ImageProcessingModel:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def predict(self, image):
        # Convert image to expected format
        image = Image.fromarray(image)
        # Process image using YOLOv5 model
        results = self.model(image)
        detections = []
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            detections.append((x1, y1, x2, y2, conf, cls))
        return detections
