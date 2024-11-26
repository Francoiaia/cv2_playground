import numpy as np
import cv2


def mask_to_obb(mask):
    """
    Convert a binary mask to an oriented bounding box (OBB) and zero out points outside the OBB.

    Args:
        mask: Binary numpy array where 1s represent the object

    Returns:
        tuple: (masked_result, box_points, (center, (width, height), angle))
            - masked_result: Binary mask with points outside OBB set to 0
            - box_points: np.array of 4 corner points of OBB
            - box_params: (center, (width, height), angle) of the OBB
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask, None, None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle
    box_params = cv2.minAreaRect(largest_contour)
    box_points = cv2.boxPoints(box_params)
    box_points = np.int0(box_points)

    # Create empty mask of same size
    result_mask = np.zeros_like(mask)

    # Fill the OBB region with 1s
    cv2.fillPoly(result_mask, [box_points], 1)

    # Mask the original image to keep only points inside OBB
    masked_result = cv2.bitwise_and(mask,result_mask)

    return masked_result, box_points, box_params


def visualize_result(original_mask, masked_result, box_points):
    """
    Helper function to visualize the results.

    Args:
        original_mask: Original binary mask
        masked_result: Mask after OBB filtering
        box_points: Corner points of the OBB
    """
    # Create RGB visualizations
    original_vis = np.stack([original_mask * 255] * 3, axis=-1).astype(np.uint8)
    result_vis = np.stack([masked_result * 255] * 3, axis=-1).astype(np.uint8)

    # Draw the OBB on both
    cv2.drawContours(original_vis, [box_points], 0, (0, 255, 0), 2)
    cv2.drawContours(result_vis, [box_points], 0, (0, 255, 0), 2)

    return original_vis, result_vis


