import os
import cv2
from ultralytics import YOLO

for file in os.listdir("image"):
    image = cv2.imread(f"image/{file}")
