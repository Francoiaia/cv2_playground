import cv2
import numpy as np
from cv2 import Mat
from numpy import ndarray, dtype
from ultralytics import YOLO
from typing import List, Tuple, Any

from segmentation.utils import calculate_kernel, check_person_on_bed


def analyze_video_frames(
        video_path: str,
        interval_seconds: int = 2,
        num_frames: int = 3,
        model_path: str = 'yolo11x-seg.pt'
) -> tuple[list[Any], Mat | ndarray]:
    """
    Extract frames from a video and perform YOLO analysis, returning the results.

    Args:
        video_path (str): Path to the input video file
        interval_seconds (int): Time interval between frames in seconds
        num_frames (int): Number of frames to extract
        model_path (str): Path to the YOLO model file

    Returns:
        List: List of YOLO results for each extracted frame
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    # Load YOLO model
    model = YOLO(model_path)

    # Initialize results list
    yolo_results = []
    frames_processed = 0
    frame_count = 0
    _, return_frame = cap.read()
    while frames_processed < num_frames:

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Perform YOLO analysis
            results = model(frame, conf=0.50)

            yolo_results.append(results[0])
            frames_processed += 1

        frame_count += 1

    cap.release()
    return yolo_results, return_frame


if __name__ == "__main__":
    video_path = "image/mandre_broom_walk_27.mp4"
    results, image = analyze_video_frames(video_path)

    interested_class = [57, 59]
    color = (0, 255, 0)  # Green color for overlay
    alpha = 0.5  # Transparency factor
    img_h, img_w = image.shape[:2]
    merged_mask = np.zeros(shape=(384, 640), dtype=np.float32)
    person_bbox = None
    person_masks = None
    person_in_image = False

    for count, frame in enumerate(results):
        for result in frame:
            if result.boxes.cls.numpy() == 0:
                person_in_image = True
                person_bbox = result.boxes.xyxyn.numpy()
                person_masks = result.masks.data.numpy().squeeze()

            r = result.cpu()
            cls = r.boxes.cls.numpy()[0]
            if cls in interested_class:
                if person_bbox is not None:
                    furniture_bbox = r.boxes.xyxyn.numpy()
                    overlap = check_person_on_bed(person_bbox=person_bbox, bed_bbox=furniture_bbox)
                    print(overlap)
                furniture_mask = r.masks.data.numpy().squeeze()

                merged_mask = cv2.bitwise_or(furniture_mask, merged_mask)
                # Create colored mask

                furniture_mask = cv2.resize(furniture_mask, (img_w, img_h),
                                            interpolation=cv2.INTER_NEAREST)

                colored_mask = np.zeros_like(image)
                colored_mask[furniture_mask > 0] = color

                overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
                cv2.imwrite(f"results/figa_{count}_{cls}.jpg", overlay)

    kernel = calculate_kernel(merged_mask, percentage=.15)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    expanded_edges = cv2.dilate(merged_mask, kernel, iterations=1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    expanded_edges = cv2.morphologyEx(expanded_edges, cv2.MORPH_CLOSE, kernel)
    kernel = calculate_kernel(expanded_edges, percentage=.05)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    merged_furniture_person = cv2.erode(expanded_edges, kernel, iterations=2)

    merged_mask = cv2.resize(merged_furniture_person, (img_w, img_h),
                             interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(image)
    colored_mask[merged_mask > 0] = color

    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    cv2.imwrite(f"results/merged_before.jpg", overlay)
