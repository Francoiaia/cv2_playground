import cv2
import numpy as np
from pydantic import BaseModel


class FurnitureMaskSchema(BaseModel):
    id_camera: str
    points: list[tuple[int, int]]
    image_width: int
    image_height: int
    class_name: str


def get_bbox_from_contour(contour_points):
    # Extract x and y coordinates
    x_coords = [point[0] for point in contour_points]
    y_coords = [point[1] for point in contour_points]

    # Find the minimum and maximum values for x and y
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Return the bounding box coordinates as (x_min, y_min, x_max, y_max)
    return min_x, min_y, max_x, max_y


def draw_bbox_on_image(frame, bboxs, output_path=None):
    if frame is None:
        raise ValueError(f"Unable to read the image at frame")
    for bbox in bboxs:
        # Draw the bounding box
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return frame


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    :param bbox1: A tuple of (x_min, y_min, x_max, y_max) for the first bounding box
    :param bbox2: A tuple of (x_min, y_min, x_max, y_max) for the second bounding box
    :return: The IoU value as a float between 0 and 1
    """
    # Calculate the (x, y) coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    bbox1_area = abs(bbox1[2] - bbox1[0]) * abs(bbox1[3] - bbox1[1])
    bbox2_area = abs(bbox2[2] - bbox2[0]) * abs(bbox2[3] - bbox2[1])

    # Calculate the Union area by using the formula:
    # Union(A,B) = A + B - Intersection(A,B)
    union_area = bbox1_area + bbox2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area != 0 else 0.0

    return iou


def calculate_overlap_percentage(bbox1, bbox2):
    """
    Calculate the overlapping ot BBOX1 in BBOX2.

    :param bbox1: A tuple of (x_min, y_min, x_max, y_max) for the first bounding box
    :param bbox2: A tuple of (x_min, y_min, x_max, y_max) for the second bounding box
    :return: return a value in percentage 0-100%
    """
    # Calculate the (x, y) coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    bbox1_area = abs(bbox1[2] - bbox1[0]) * abs(bbox1[3] - bbox1[1])

    overlap_percentage = (intersection_area / bbox1_area) * 100

    return overlap_percentage


def draw_bbox_and_overlap(frame, bbox1, bbox2, alpha=0.2):
    # Unpack the bounding box coordinates
    (x1, y1, x2, y2) = bbox1  # First bbox
    (x3, y3, x4, y4) = bbox2  # Second bbox
    overlay = frame.copy()

    # Draw the two bounding boxes on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)  # Blue box

    # Calculate the overlapping box (intersection)
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    # Check if there's an overlap
    if x_right > x_left and y_bottom > y_top:
        # Draw the overlapping region in a different color (e.g., red)
        cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (180, 180, 180), -1)  # Filled red
        # Apply alpha blending to the overlapping area
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame


def calculate_percentage_in_mask(mask_points, bbox):
    # Convert bbox from (x1, y1, x2, y2) to (x, y, width, height)
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Create a blank mask image
    bbox_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

    # Convert mask points to a numpy array and shift them relative to the bounding box's top-left corner

    # Draw the polygon mask on the blank image within the bounding box
    cv2.fillPoly(bbox_mask, [mask_points], 255)

    # Calculate the area of the bounding box
    bbox_area = bbox_width * bbox_height

    # Calculate the area covered by the mask within the bounding box
    mask_area = np.sum(bbox_mask > 0)  # Count non-zero pixels in the mask
    # couch_area = cv2.contourArea(contours[0])

    # Calculate the percentage of the bounding box area that is in the mask
    percentage_in_mask = (mask_area / bbox_area) * 100 if bbox_area > 0 else 0

    return percentage_in_mask


def calculate_bbox_mask_overlap(frame, bbox, mask_points, alpha=0.5):
    """
    Calculate the percentage of a bounding box that overlaps with a mask defined by points,
    and visualize the result on the input frame with an alpha overlay.

    :param frame: The input frame (numpy array) to draw the visualization on
    :param bbox: A tuple of (x1, y1, x2, y2) representing the bounding box
    :param mask_points: A list of (x, y) tuples representing the mask polygon
    :param alpha: The transparency of the overlap area (0.0 to 1.0)
    :return: The frame with visualizations and the overlap percentage
    """
    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Create a blank image for mask and overlap calculation
    img_height, img_width = frame.shape[:2]
    img = np.zeros((img_height, img_width), dtype=np.uint8)

    # Draw the mask polygon on the blank image
    mask_points_array = np.array(mask_points, dtype=np.int32)
    cv2.fillPoly(img, [mask_points_array], 255)

    # Create the bounding box mask
    bbox_mask = np.zeros_like(img)
    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)

    # Calculate the overlap
    overlap = cv2.bitwise_and(img, bbox_mask)
    overlap_area = cv2.countNonZero(overlap)
    bbox_area = (x2 - x1) * (y2 - y1)

    # Calculate the percentage
    overlap_percentage = (overlap_area / bbox_area) * 100

    # Create a copy of the frame for overlay
    overlay = frame.copy()

    # Visualize the mask polygon on the overlay
    cv2.polylines(overlay, [mask_points_array], True, (0, 255, 0), 2)


    # Visualize the overlap area on the overlay with alpha transparency
    overlap_coords = np.where(overlap > 0)
    overlay[overlap_coords[0], overlap_coords[1]] = [0, 0, 255]  # Red color for overlap

    # Combine the original frame and overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame, overlap_percentage
def calculate_bbox_mask_overlap_cv2(frame, bbox, mask_points, alpha=0.5):
    """
    Calculate the percentage of a bounding box that overlaps with a mask defined by points,
    using contour area for calculations. Visualize the result on the input frame with an alpha overlay.

    :param frame: The input frame (numpy array) to draw the visualization on
    :param bbox: A tuple of (x1, y1, x2, y2) representing the bounding box
    :param mask_points: A list of (x, y) tuples representing the mask polygon
    :param alpha: The transparency of the overlap area (0.0 to 1.0)
    :return: The frame with visualizations and the overlap percentage
    """
    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Create a blank image for mask and overlap calculation
    img_height, img_width = frame.shape[:2]
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    bbox_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Draw the mask polygon and bounding box on separate masks
    mask_points_array = np.array(mask_points, dtype=np.int32)
    cv2.fillPoly(mask, [mask_points_array], 255)
    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)

    # Calculate the overlap
    overlap = cv2.bitwise_and(mask, bbox_mask)

    # Find contours for mask and overlap
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlap_contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate areas using contour area
    mask_area = sum(cv2.contourArea(contour) for contour in mask_contours)
    overlap_area = sum(cv2.contourArea(contour) for contour in overlap_contours)
    bbox_area = (x2 - x1) * (y2 - y1)

    # Calculate the percentage
    overlap_percentage = (overlap_area / bbox_area) * 100

    # Create a copy of the frame for overlay
    overlay = frame.copy()

    # Visualize the mask polygon on the overlay
    cv2.polylines(overlay, [mask_points_array], True, (0, 255, 0), 2)

    # Visualize the bounding box on the overlay
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Visualize the overlap area on the overlay with alpha transparency
    overlap_coords = np.where(overlap > 0)
    overlay[overlap_coords[0], overlap_coords[1]] = [0, 0, 255]  # Red color for overlap

    # Combine the original frame and overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame, overlap_percentage