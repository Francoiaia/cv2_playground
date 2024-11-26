import random
import time

import cv2
import numpy as np
import torch


def get_camera_frame(source: str, frame_number: int = 10, max_read_miss: int = 20):
    cap = cv2.VideoCapture(source)
    time.sleep(2)

    if not cap.isOpened():
        return None

    read_miss = 0
    i = 0
    frame = None
    while i <= frame_number:
        success, frame = cap.read()

        if success:
            i += 1
            read_miss = 0
        else:
            read_miss += 1

        if read_miss > max_read_miss:
            return None

    cap.release()
    # cv2.imwrite("frame%d.jpg" % frame_number, frame)
    return frame


def are_masks_close(mask1, mask2, threshold=0.15):  # threshold as percentage of normalized space
    # Calculate centroids (will be in normalized coordinates)
    centroid1 = np.mean(mask1, axis=0)
    centroid2 = np.mean(mask2, axis=0)

    # Calculate Euclidean distance between centroids (in normalized space)
    distance = np.sqrt(np.sum((centroid1 - centroid2) ** 2))

    # Since we're in normalized space (0-1), the distance will also be normalized
    return {
        'are_close': distance < threshold,
        'distance': distance
    }


def draw_points_with_lines(frame, points):
    points = np.array(points).astype(int)

    # Draw lines between points
    cv2.polylines(frame, [points],
                  isClosed=True,  # True if you want to connect last point to first
                  color=(0, 255, 0),  # Green color
                  thickness=2)

    # Draw points
    for x, y in points:
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    return frame


def display_mask_on_image(image, mask, alpha=0.5, color=(0, 0, 255)):
    if mask.shape[0] == 1:
        mask = mask.squeeze(0)

    # 1. Display the binary mask
    # Convert to uint8 (0 or 255)
    binary_display = (mask * 255).astype(np.uint8)
    cv2.imwrite('mask.jpg', binary_display)
    return
    # 2. Create a colored overlay
    # Create a colored mask (e.g., red overlay)
    colored_mask = np.zeros_like(binary_display)
    colored_mask[mask == 1] = 255  # Red color where mask is 1

    # Overlay on original image
    alpha = 0.5  # Transparency factor
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    # Display the overlay
    cv2.imshow('Overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_person_on_bed(person_bbox, bed_bbox):
    """
    Check if person is on bed using normalized bounding boxes (0 to 1)

    Args:
        person_bbox: [x1, y1, x2, y2] of person in normalized coordinates
        bed_bbox: [x1, y1, x2, y2] of bed in normalized coordinates

    Returns:
        overlap_ratio: float between 0 and 1
    """
    # Get the bounding boxes
    person_bbox = person_bbox[0]
    bed_bbox = bed_bbox[0]

    # Calculate intersection in normalized coordinates
    x_left = max(person_bbox[0], bed_bbox[0])
    y_top = max(person_bbox[1], bed_bbox[1])
    x_right = min(person_bbox[2], bed_bbox[2])
    y_bottom = min(person_bbox[3], bed_bbox[3])

    # If there's no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate areas in normalized space
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])

    # Calculate overlap ratio
    overlap_ratio = intersection_area / person_area if person_area > 0 else 0.0

    return overlap_ratio


def calculate_kernel(mask, percentage=0.12):
    white_pixels = np.sum(mask > 0)
    kernel_size = int(np.sqrt(white_pixels) * percentage)  # 10% of sqrt of object size
    kernel_size = max(3, kernel_size)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return kernel


def get_obb_from_mask(mask):
    """
    Calculate Oriented Bounding Box from a binary mask

    Args:
        mask: Binary mask (numpy array where pixels are 0 or 1)

    Returns:
        box: Array of 4 points representing the oriented bounding box corners
        xyxyn: Axis-aligned bounding box in normalized [x_min, y_min, x_max, y_max] format
        angle: Rotation angle of the box
    """
    height, width = mask.shape[:2]

    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    if len(contours) == 1:
        # Get the single contour
        cnt = contours[0]
        points = cnt.reshape(-1, 2)
    else:
        # Combine all contours
        points = np.vstack([cnt.reshape(-1, 2) for cnt in contours])

    # Calculate the minimum area oriented bounding box
    rect = cv2.minAreaRect(points)

    # Ensure the rectangle is oriented along the longer side
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90

    # Create new rect with adjusted orientation
    rect = ((cx, cy), (w, h), angle)

    box = cv2.boxPoints(rect)
    box = np.int0(box)  # Convert to integer coordinates

    x1, y1 = box[0]
    x2, y2 = box[1]
    x3, y3 = box[2]
    x4, y4 = box[3]

    # Calculate min and max for x and y
    x1 = min(x1, x2, x3, x4)
    x2 = max(x1, x2, x3, x4)
    y1 = min(y1, y2, y3, y4)
    y2 = max(y1, y2, y3, y4)

    # Normalize coordinates
    x1_norm = x1 / width
    y1_norm = y1 / height
    x2_norm = x2 / width
    y2_norm = y2 / height

    xyxyn = np.array([[x1_norm, y1_norm, x2_norm, y2_norm]])

    return box, [[x1, y1, x2, y2]], xyxyn, angle


def draw_bboxes(image, bboxes, norm=True):
    """
    Draw multiple bounding boxes on an image

    Args:
        image: Image to draw on (numpy array)
        bboxes: List of bounding boxes in xyxy format
        colors: Optional list of colors for each box
        thickness: Line thickness
        font_scale: Size of the font for the labels

    Returns:
        image: Image with drawn bounding boxes
    """
    height, width = image.shape[:2]
    image = image.copy()

    for bbox in bboxes:
        # Handle both numpy arrays and lists
        bbox = bbox[0]
        if norm:
            # Denormalize coordinates
            x1 = int(float(bbox[0]) * width)
            y1 = int(float(bbox[1]) * height)
            x2 = int(float(bbox[2]) * width)
            y2 = int(float(bbox[3]) * height)
        else:
            x1 = int(float(bbox[0]))
            y1 = int(float(bbox[1]))
            x2 = int(float(bbox[2]))
            y2 = int(float(bbox[3]))
        # Draw the rectangle
        image = cv2.rectangle(image, (x1, y1), (x2, y2),
                              color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                              thickness=2)

    return image


def crop_person_to_couch_outline(person_mask, couch_mask, merged, image):
    person_contours, _ = cv2.findContours(person_mask.astype(np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    couch_contours, _ = cv2.findContours(couch_mask.astype(np.uint8),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    merged_contours, _ = cv2.findContours(merged.astype(np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    image = cv2.resize(image, (couch_mask.shape[1], couch_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    img_with_obb_contours = image.copy()
    # for cont in couch_contours:
    #     img_with_obb_contours = cv2.drawContours(img_with_obb_contours, [cont], 0, (0, 0, 255), 2)
    # for cont in person_contours:
    #     img_with_obb_contours = cv2.drawContours(img_with_obb_contours, [cont], 0, (0, 0, 255), 2)
    for cont in merged_contours:
        img_with_obb_contours = cv2.drawContours(img_with_obb_contours, [cont], 0, (0, 0, 255), 2)

    return img_with_obb_contours


def mask_within_bbox(mask, bbox):
    height, width = mask.shape
    bbox = bbox[0]
    # Convert bbox coordinates to integers
    x_min = int(bbox[0] * width)
    y_min = int(bbox[1] * height)
    x_max = int(bbox[2] * width)
    y_max = int(bbox[3] * height)

    # Create a copy of the mask
    masked = mask.copy()

    # Set all elements outside the bbox to 0
    masked[:y_min, :] = 0  # Above the bbox
    masked[y_max:, :] = 0  # Below the bbox
    masked[:, :x_min] = 0  # Left of the bbox
    masked[:, x_max:] = 0  # Right of the bbox

    return masked


def filter_mask_by_bbox(mask, bbox_points):
    """
    Filter a binary mask using a quadrilateral defined by 4 points

    Args:
        mask: 2D numpy array with binary values (0 or 1, or True/False)
        bbox_points: List of 4 [x,y] coordinates

    Returns:
        Filtered mask with same shape as input, zeros outside bbox
    """
    # Create a polygon mask
    polygon_mask = np.zeros_like(mask, dtype=np.uint8)

    # Convert bbox_points to the format needed by fillPoly
    points = np.array(bbox_points, dtype=np.int32)

    # Fill the polygon
    cv2.fillPoly(polygon_mask, [points], 1)

    # Apply the polygon mask to the original mask
    filtered_mask = mask * polygon_mask.astype(float)

    return filtered_mask


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on an image.

    Args:
        image (numpy.ndarray): Input image
        bbox (tuple/list): Bounding box coordinates in (x1, y1, x2, y2) format
        color (tuple): BGR color for the bbox (default: green)
        thickness (int): Line thickness (default: 2)
    Returns:
        numpy.ndarray: Image with drawn bbox
    """
    # Make sure we're working with integers
    x1, y1, x2, y2 = map(int, bbox[0])

    # Draw the bounding box
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image


def print_mask_onto_image(image=None, merged_furniture_person=None, color=None, alpha=None, save_name=None):
    img_h, img_w = image.shape[:2]

    merged_mask = cv2.resize(merged_furniture_person, (img_w, img_h),
                             interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(image)
    colored_mask[merged_mask > 0] = color

    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    cv2.imwrite(f"results/{save_name}.jpg", overlay)


def do_morphology(mask_to_morph, percentage_to_dilate=.12, percentage_to_close=.12, percentage_to_erode=.12):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    # kernel = calculate_kernel(mask_to_morph, percentage_to_dilate)
    expanded_edges = cv2.dilate(mask_to_morph, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    # kernel = calculate_kernel(expanded_edges, percentage_to_close)
    expanded_edges = cv2.morphologyEx(expanded_edges, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # kernel = calculate_kernel(mask_to_morph, percentage_to_erode)
    mask_to_morph = cv2.erode(expanded_edges, kernel, iterations=2)
    return mask_to_morph


import numpy as np


def calculate_mask_overlap(mask1, mask2):
    """
    Calculate various overlap metrics between two masks.

    Args:
        mask1: First mask (numpy array)
        mask2: Second mask (numpy array)

    Returns:
        dict: Dictionary containing different overlap metrics
    """
    # Ensure masks are binary
    mask1_binary = (mask1 > 0).astype(np.float32)
    mask2_binary = (mask2 > 0).astype(np.float32)

    # Calculate intersection and union
    intersection = np.logical_and(mask1_binary, mask2_binary)
    union = np.logical_or(mask1_binary, mask2_binary)

    # Count pixels
    intersection_pixels = np.sum(intersection)
    union_pixels = np.sum(union)
    mask1_pixels = np.sum(mask1_binary)
    mask2_pixels = np.sum(mask2_binary)

    # Calculate various metrics
    metrics = {
        # IoU (Intersection over Union)
        'iou': float(intersection_pixels) / float(union_pixels) if union_pixels > 0 else 0.0,

        # Dice coefficient (F1 score)
        'dice': 2.0 * intersection_pixels / (mask1_pixels + mask2_pixels) if (mask1_pixels + mask2_pixels) > 0 else 0.0,

        # Percentage of mask1 overlapped by mask2
        'mask1_overlap': intersection_pixels / mask1_pixels if mask1_pixels > 0 else 0.0,

        # Percentage of mask2 overlapped by mask1
        'mask2_overlap': intersection_pixels / mask2_pixels if mask2_pixels > 0 else 0.0,

        # Raw pixel counts
        'intersection_pixels': int(intersection_pixels),
        'union_pixels': int(union_pixels),
        'mask1_pixels': int(mask1_pixels),
        'mask2_pixels': int(mask2_pixels)
    }

    return metrics


def visualize_overlap(mask1, mask2, title="Mask Overlap Visualization"):
    """
    Create a visualization of mask overlap.
    Red: Mask1 only
    Blue: Mask2 only
    Purple: Overlap

    Args:
        mask1: First mask
        mask2: Second mask
        title: Title for the visualization

    Returns:
        numpy array: RGB visualization image
    """
    # Ensure masks are binary
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)

    # Create RGB image (default black background)
    height, width = mask1.shape[:2]
    viz = np.zeros((height, width, 3), dtype=np.uint8)

    # Red for mask1 only
    viz[mask1_binary == 1, 0] = 255

    # Blue for mask2 only
    viz[mask2_binary == 1, 2] = 255

    # Purple for overlap (red + blue)
    overlap = np.logical_and(mask1_binary, mask2_binary)
    viz[overlap, 0] = 255  # Red channel
    viz[overlap, 2] = 255  # Blue channel

    return viz

