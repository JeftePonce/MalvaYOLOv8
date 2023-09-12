import cv2
import numpy as np

from ultralytics import YOLO

# Load a model
model = YOLO('TrainModelMalva.pt')  

image = cv2.imread('1.jpg')

detections = model.predict(image)

print(detections)