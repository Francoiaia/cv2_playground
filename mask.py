# Main execution
from pydantic import BaseModel
import cv2
import numpy as np

from utils import get_bbox_from_contour, FurnitureMaskSchema, draw_bbox_on_image, calculate_iou, draw_bbox_and_overlap, \
    calculate_overlap_percentage, calculate_percentage_in_mask, calculate_bbox_mask_overlap, \
    calculate_bbox_mask_overlap_cv2

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 30)
fontScale = 1
fontColor = (255, 255, 255)
thickness = 1
lineType = 2

if __name__ == "__main__":
    from ultralytics import YOLO

    image = cv2.imread("frame.jpg")
    # 56: chair, 57: couch, 59: bed
    interested_class = [57, 59]

    model = YOLO(f"yolov8l-seg.pt")
    pose_model = YOLO("yolov8l-pose.pt")
    models = YOLO("yolov8l.pt")
    results = model(image, conf=0.5, save=True)

    result = results[0]

    height, width, _ = image.shape

    furniture_masks = []
    i = 0
    for r in result:
        r = r.cpu()
        cls = r.boxes.cls.numpy()[0]
        if cls in interested_class:
            print(r)
            furniture = FurnitureMaskSchema(
                id_camera="aa",
                class_name=r.names[cls],
                points=r.masks.xy[0].astype(int).tolist(),
                image_height=height,
                image_width=width,
            )
            furniture_masks.append(furniture)
            # couch_bbox = get_bbox_from_contour(furniture.points)
            couch_bbox = r.boxes.xyxy.cpu().numpy().astype(int)[0]
            couch_mask = r.masks.xy[0]
            contours, hierarchy = cv2.findContours(r.masks.xy[0].astype(np.uint8), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            i = i + 1

    print(furniture_masks)

    video_path = "couch_iaia_27.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        cv2.drawContours(frame, [r.masks.xy[0].astype(np.int32)], -1, 255, 2)

        if success:
            # Run YOLOv8 inference on the frame
            results = pose_model(frame, conf=0.9)
            pose_bbox = results[0].boxes.xyxy.cpu().numpy().astype(int)
            if pose_bbox.any():
                x1, y1, x2, y2 = pose_bbox[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)

                frame, overlap_area = calculate_bbox_mask_overlap_cv2(frame, pose_bbox[0], r.masks.xy[0].astype(np.int32))

                cv2.putText(frame, f"{overlap_area}",
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                cv2.imshow("YOLOv8 Inference", frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
