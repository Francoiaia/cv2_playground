import os
import random

import cv2
import numpy as np
from sympy import im
from ultralytics import YOLO

from segmentation.utils import check_person_on_bed, calculate_kernel, crop_person_to_couch_outline, mask_within_bbox, \
    get_obb_from_mask, filter_mask_by_bbox

model = YOLO("yolo11l-seg.pt")  # load an official model
for file in os.listdir("image"):
    filename = file.removesuffix(".jpg")

    image = cv2.imread(f"image/{file}")

    # image = cv2.imread("image/photo_3_2024-11-18_11-22-28.jpg")
    result = model(image, show=False, save=False, conf=0.5)
    interested_class = [57, 59]
    result = result[0]
    person_bbox = None
    person_masks = None
    person_in_image = False
    for res in result:
        if res.boxes.cls.numpy() == 0:
            person_in_image = True
            person_bbox = res.boxes.xyxyn.numpy()
            person_masks = res.masks.data.numpy()

    for r in result:
        r = r.cpu()
        cls = r.boxes.cls.numpy()[0]
        if cls in interested_class:
            furniture_mask = r.masks.data.numpy().squeeze()

            if person_bbox is not None:
                person_masks = person_masks.squeeze()
                furniture_bbox = r.boxes.xyxyn.numpy()
                overlap = check_person_on_bed(person_bbox=person_bbox, bed_bbox=furniture_bbox)

                print(f"{file}; bbox {overlap}")
                if person_in_image:
                    if overlap > .95:
                        print("Unifiy mask")

                        # To draw the box (for visualization):

                        #
                        # img_with_box = cv2.drawContours(ima, [box], 0, (0, 255, 0), 2)
                        # img_with_box = cv2.drawContours(img_with_box, [box_2], 0, (0, 255, 0), 2)
                        # cv2.imshow("obb", img_with_box)
                        # cv2.waitKey(0)
                        # continue
                        merged_furniture_person = cv2.bitwise_or(person_masks, furniture_mask)

                        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                        kernel = calculate_kernel(merged_furniture_person)
                        expanded_edges = cv2.dilate(merged_furniture_person, kernel, iterations=1)

                        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                        expanded_edges = cv2.morphologyEx(expanded_edges, cv2.MORPH_CLOSE, kernel)

                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
                        merged_furniture_person = cv2.erode(expanded_edges, kernel, iterations=2)

                        contours, hierarchy = cv2.findContours(merged_furniture_person.astype(np.uint8),
                                                               cv2.RETR_EXTERNAL,
                                                               cv2.CHAIN_APPROX_SIMPLE)

                        result_mask = crop_person_to_couch_outline(person_masks, furniture_mask,
                                                                   merged_furniture_person, image)
                        # cv2.imwrite(f"image/results/{filename}_mask.jpg",
                        #             (merged_furniture_person * 255).astype(np.uint8))  # overlay = image.copy()
                        color = (0, 255, 0)  # Green color for overlay
                        alpha = 0.5  # Transparency factor
                        img_h, img_w = image.shape[:2]

                        merged_furniture_person = cv2.resize(merged_furniture_person, (img_w, img_h),
                                                             interpolation=cv2.INTER_NEAREST)
                        # Create colored mask
                        colored_mask = np.zeros_like(image)
                        colored_mask[merged_furniture_person > 0] = color

                        # Combine image and colored mask
                        overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
                        cv2.imwrite(f"image/results/{filename}_overlay_result.jpg", overlay)

                    elif overlap > .1:

                        box, xyxy, xyxyn, angle = get_obb_from_mask(furniture_mask)
                        furniture_mask = filter_mask_by_bbox(cv2.bitwise_or(furniture_mask, person_masks), box)

                        img_with_obb_contours = cv2.drawContours(
                            cv2.resize(image, (furniture_mask.shape[1], furniture_mask.shape[0])), [box], 0,
                            (0, 0, 255), 2)

                        # furniture_mask = mask_within_bbox(bbox=furniture_bbox,
                        #                                   mask=cv2.bitwise_or(furniture_mask, person_masks))
                        #
                        kernel = calculate_kernel(furniture_mask)
                        merged_furniture_person = cv2.dilate(furniture_mask, kernel, iterations=2)
                        merged_furniture_person = cv2.morphologyEx(merged_furniture_person, cv2.MORPH_CLOSE, kernel)
                        merged_furniture_person = cv2.erode(merged_furniture_person, kernel, iterations=2)
                        merged_furniture_person = cv2.dilate(merged_furniture_person, np.ones((5, 5)), iterations=2)
                        merged_furniture_person = mask_within_bbox(bbox=furniture_bbox,
                                                                   mask=merged_furniture_person)

                        print("morfologia più aggressiva")
                    else:
                        kernel = calculate_kernel(furniture_mask)
                        expanded_edges = cv2.dilate(furniture_mask, kernel, iterations=3)
                        expanded_edges = cv2.morphologyEx(expanded_edges, cv2.MORPH_CLOSE, kernel)
                        expanded_edges = cv2.erode(expanded_edges, kernel, iterations=3)
                        merged_furniture_person = cv2.dilate(expanded_edges, np.ones((5, 5)), iterations=3)

                        print("Segmentation normale")
                    color = (0, 255, 0)  # Green color for overlay
                    alpha = 0.5  # Transparency factor
                    img_h, img_w = image.shape[:2]
                    merged_furniture_person = cv2.resize(merged_furniture_person, (img_w, img_h),
                                                         interpolation=cv2.INTER_NEAREST)
                    # Create colored mask
                    colored_mask = np.zeros_like(image)
                    colored_mask[merged_furniture_person > 0] = color

                    # Combine image and colored mask
                    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
                    cv2.imwrite(f"image/results/{filename}_overlay_result.jpg", overlay)
