import os
from uuid import UUID

import cv2
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

from segmentation.utils import check_person_on_bed, calculate_kernel, mask_within_bbox, \
    get_obb_from_mask, filter_mask_by_bbox, draw_bbox




# version = "v8x"
# version = "v8l"
# version = "11l"
# version = "11x"
versions = ["11l"]
# versions = ["v8x", "v8l", "11l", "11x"]
# for version in versions:
for version in versions:
    model = YOLO(f"yolo{version}-seg.pt")  # load an official model

    for file in os.listdir("image/jpg"):

        filename = file.removesuffix(".jpg")

        image = cv2.imread(f"image/jpg/{file}")
        # image = cv2.imread("image/photo_3_2024-11-18_11-22-28.jpg")
        result = model(image, show=False, save=False, conf=0.59)

        result = result[0]
        filedio = f"/results/{version}/{filename}_yolo_{version}.jpg"
        result.save(filename=filedio)  # save to disk

        print(f"{file}")

        interested_class = [57, 59]
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

                    print(f"bbox {overlap}")
                    if person_in_image:
                        if overlap > .95:
                            print("Unifiy mask")

                            merged_furniture_person = cv2.bitwise_or(person_masks, furniture_mask)

                            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                            kernel = calculate_kernel(merged_furniture_person)
                            expanded_edges = cv2.dilate(merged_furniture_person, kernel, iterations=1)

                            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                            expanded_edges = cv2.morphologyEx(expanded_edges, cv2.MORPH_CLOSE, kernel)

                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
                            merged_furniture_person = cv2.erode(expanded_edges, kernel, iterations=2)

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
                            cv2.imwrite(f"results/{version}/{filename}_overlay_result_{version}.jpg", overlay)

                        elif overlap > .1:
                            box, xyxy, xyxyn, angle = get_obb_from_mask(furniture_mask)
                            furniture_mask = filter_mask_by_bbox(cv2.bitwise_or(furniture_mask, person_masks), box)

                            # image = cv2.drawContours(
                            #     cv2.resize(image, (furniture_mask.shape[1], furniture_mask.shape[0])), [box], 0,
                            #     (0, 0, 255), 2)
                            image = draw_bbox(image, r.boxes.xyxy.numpy(), )
                            # furniture_mask = mask_within_bbox(bbox=furniture_bbox,
                            #                                   mask=cv2.bitwise_or(furniture_mask, person_masks))

                            kernel = calculate_kernel(furniture_mask)
                            merged_furniture_person = cv2.dilate(furniture_mask, kernel, iterations=2)
                            merged_furniture_person = cv2.morphologyEx(merged_furniture_person, cv2.MORPH_CLOSE, kernel)
                            merged_furniture_person = cv2.erode(merged_furniture_person, kernel, iterations=2)
                            merged_furniture_person = cv2.dilate(merged_furniture_person, np.ones((5, 5)), iterations=2)

                            merged_furniture_person = mask_within_bbox(bbox=furniture_bbox,
                                                                       mask=merged_furniture_person)

                            print("morfologia più aggressiva")
                        else:
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                            expanded_edges = cv2.dilate(furniture_mask, kernel, iterations=1)

                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                            expanded_edges = cv2.morphologyEx(expanded_edges, cv2.MORPH_CLOSE, kernel)

                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
                            merged_furniture_person = cv2.erode(expanded_edges, kernel, iterations=2)

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
                        cv2.imwrite(f"results/{version}/{filename}_overlay_result_{version}.jpg", overlay)
                else:
                    kernel = calculate_kernel(furniture_mask, percentage=.15)

                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
                    expanded_edges = cv2.dilate(furniture_mask, kernel, iterations=1)

                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
                    expanded_edges = cv2.morphologyEx(expanded_edges, cv2.MORPH_CLOSE, kernel)
                    kernel = calculate_kernel(furniture_mask, percentage=.05)

                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
                    merged_furniture_person = cv2.erode(expanded_edges, kernel, iterations=2)
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
                    cv2.imwrite(f"results/{version}/{filename}_overlay_result_{version}.jpg", overlay)
                    print("Persona non in immagine")
