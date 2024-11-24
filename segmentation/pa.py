ima = cv2.resize(image, (person_masks.shape[1], person_masks.shape[0]),
                 interpolation=cv2.INTER_NEAREST)

# NORMAL BBOX
overlap = check_person_on_bed(person_bbox=person_bbox, bed_bbox=furniture_bbox)
img_with_nbbox = draw_bboxes(ima.copy(), [person_bbox, furniture_bbox], True)
cv2.imwrite(f"image/results/{filename}_nbb.jpg", img_with_nbbox)

# ORIENTED BBOX
normalized_box, angle, normalized_center, normalized_dims = get_obb_from_mask(person_masks)
normalized_box_2, angle_2, normalized_center_2, normalized_dims_2 = get_obb_from_mask(furniture_mask)
overlap_obb = check_person_on_bed_centered(person_bbox=normalized_box, bed_bbox=furniture_bbox)
img_with_obbox = draw_bboxes_2(ima.copy(),
                               [normalized_box],
                               "cxcywh",
                               False)
# [bbox_xyxyn, furniture_bbox, box_xyxy])
cv2.imwrite(f"image/results/{filename}_obb.jpg", img_with_obbox)

# ORIENTED BBOX WITH CONTOUR
img_with_obb_contours = cv2.drawContours(ima.copy(), [box], 0, (0, 0, 255), 2)
img_with_obb_contours = cv2.drawContours(img_with_obb_contours, [box_2], 0, (0, 0, 255), 2)
cv2.imwrite(f"image/results/{filename}_obb_contours.jpg", img_with_obb_contours)
