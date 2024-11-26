from typing import List, Any

import cv2
import numpy as np
from cv2 import Mat
from numpy import ndarray
from ultralytics import YOLO

from segmentation.ultra_utils import mask_to_obb
from segmentation.utils import check_person_on_bed, mask_within_bbox, do_morphology, print_mask_onto_image


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
    name_list = [
        # "figio_fall_1.mp4",
        "mandre_broom_walk_27.mp4"
        # "figio_walk_0.mp4",
        # "lanza_broom_walk_51.mp4",
        # "iaia_bed.mp4",
    ]
    for name in name_list:
        video_path = f"image/{name}"
        print(video_path)
        results, image = analyze_video_frames(video_path)

        interested_class = [57, 59]
        color = (0, 255, 0)  # Green color for overlay
        alpha = 0.5  # Transparency factor
        furniture_mask_between_frames = np.zeros(shape=(384, 640), dtype=np.float32)
        person_mask_between_frames = np.zeros(shape=(384, 640), dtype=np.float32)
        person_bbox = None
        person_masks = None
        person_in_image = False
        counter = 0
        for frame in results:

            for result in frame:
                if result.boxes.cls.numpy() == 0:
                    person_in_image = True
                    person_bbox = result.boxes.xyxyn.numpy()
                    person_masks = result.masks.data.numpy().squeeze()
            frame.save(f"{video_path}_{counter}_yolo.jpg")
            cv2.imwrite(f"{video_path}_{counter}_clean.jpg", image)
            counter += 1
            for result in frame:
                r = result.cpu()
                cls = r.boxes.cls.numpy()[0]
                if cls in interested_class:
                    furniture_mask = r.masks.data.numpy().squeeze()

                    if person_bbox is not None:
                        furniture_bbox = r.boxes.xyxyn.numpy()

                        overlap = check_person_on_bed(person_bbox=person_bbox, bed_bbox=furniture_bbox)
                        # print(overlap)
                        if overlap > 120000:

                            merged_furniture_person = cv2.bitwise_or(person_masks, furniture_mask)

                            merged_furniture_person = mask_within_bbox(bbox=furniture_bbox,
                                                                       mask=merged_furniture_person)

                            mask_limed_bbox = cv2.resize(
                                merged_furniture_person,
                                (furniture_mask_between_frames.shape[1], furniture_mask_between_frames.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

                            furniture_mask_between_frames = cv2.bitwise_or(mask_limed_bbox,
                                                                           furniture_mask_between_frames)
                        else:
                            # merged_furniture_person = cv2.bitwise_or(person_masks, furniture_mask)

                            # mask_limed_bbox = mask_within_bbox(bbox=furniture_bbox,
                            #                                    mask=furniture_mask)
                            kernel = np.ones((15, 15), np.uint8)
                            filled_mask = cv2.morphologyEx(furniture_mask, cv2.MORPH_DILATE, kernel)
                            mask_limed_bbox, box_points, box_params = mask_to_obb(mask=filled_mask)
                            print_mask_onto_image(image, mask_limed_bbox, color, alpha,
                                                  f"{name}_{counter}_morph")

                            mask_limed_bbox = cv2.resize(mask_limed_bbox.astype(np.float32), (
                                furniture_mask_between_frames.shape[1], furniture_mask_between_frames.shape[0]),
                                                         interpolation=cv2.INTER_NEAREST)

                            furniture_mask_between_frames = cv2.bitwise_or(mask_limed_bbox,
                                                                           furniture_mask_between_frames)

                            person_limed_masks = mask_within_bbox(bbox=furniture_bbox,
                                                                  mask=person_masks)

                            person_mask_between_frames = cv2.bitwise_or(person_limed_masks, person_mask_between_frames)

                            # cv2.imshow('Binary Image', merged_mask_between_frames.astype(np.uint8) * 255)
                            # cv2.waitKey(0)

                    else:
                        print("not person")

        full_couch = cv2.bitwise_or(person_mask_between_frames, furniture_mask_between_frames)

        kernel = np.ones((5, 5), np.uint8)
        filled_mask = cv2.morphologyEx(full_couch, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(f"{name}_full_couch.jpg", (filled_mask * 255).astype(np.uint8))

        cv2.imwrite(f"{name}_person_mask_between_frames.jpg", (person_mask_between_frames * 255).astype(np.uint8))

        cv2.imwrite(f"{name}_furniture_mask_between_frames.jpg", (furniture_mask_between_frames * 255).astype(np.uint8))
        # overlap = calculate_mask_overlap(full_couch, person_mask_between_frames)
        # print(overlap)
        #
        # overlap = calculate_mask_overlap(full_couch, furniture_mask_between_frames)
        # print(overlap)
        # overlap = calculate_mask_overlap(person_mask_between_frames, furniture_mask_between_frames)
        # print(overlap)

        merged_mask_between_frames_morph = do_morphology(furniture_mask_between_frames,
                                                         percentage_to_dilate=.15,
                                                         percentage_to_erode=.5)
        # print_mask_onto_image(image, merged_mask_between_frames, color, alpha, f"{name}_no_morph")
        print_mask_onto_image(image, merged_mask_between_frames_morph, color, alpha, f"{name}_morph")
        # # cv2.imshow('Binary Image', merged_mask_between_frames.astype(np.uint8) * 255)
        # # cv2.waitKey(0)
