# Main execution
from pydantic import BaseModel
import cv2

from utils import get_bbox_from_contour, FurnitureMaskSchema, draw_bbox_on_image, calculate_iou, draw_bbox_and_overlap, \
    calculate_overlap_percentage

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 30)
fontScale = 1
fontColor = (255, 255, 255)
thickness = 1
lineType = 2

if __name__ == "__main__":
    from ultralytics import YOLO

    image = cv2.imread("laba_slow_fall_0 - frame at 0m0s.jpg")
    # 56: chair, 57: couch, 59: bed
    interested_class = [57, 59]

    model = YOLO(f"yolov8l-seg.pt")
    pose_model = YOLO("yolov8l-pose.pt")
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
            couch_bbox = get_bbox_from_contour(furniture.points)
            # draw_bbox_on_image(image_path, get_bbox_from_contour(furniture.points),
            #                    output_path=f"output_with_bbox_{i}.jpg")
            i = i + 1
    print(furniture_masks)

    video_path = "laba_slow_fall_0.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = pose_model(frame)
            pose_bbox = results[0].boxes.xyxy.cpu().numpy().astype(int)[0]
            frame = draw_bbox_and_overlap(frame, pose_bbox, couch_bbox)
            # iou = calculate_iou(pose_bbox, couch_bbox)
            perc = calculate_overlap_percentage(pose_bbox, couch_bbox)
            cv2.putText(frame, f"{perc}",
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
