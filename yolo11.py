from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-seg.pt")  # load an official model

# Run batched inference on a list of images
results = model("black-man-sleeping-bed-eye-mask-black-man-sleeping-bed-eye-mask-109710349.webp", show=True)  # predict on an image
