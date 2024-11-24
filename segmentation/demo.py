import cv2
from ultralytics import YOLO
from utils import display_mask_on_image, draw_points_with_lines

# Load a model
model = YOLO("yolo11x-seg.pt")  # load an official model

if __name__ == '__main__':
    # camera_frame = get_camera_frame(
    #     "rtsp://admin:admin@192.168.1.43:554/live/ch1?token=d2abaa37a7c3db1137d385e1d8c15fd2",
    #     10)

    image = cv2.imread("image/frame10.jpg")
    results = model(image, show=True, save=False, conf=0.5)  # predict on an image

    interested_class = [57, 59]

    result = results[0]
    # print(are_masks_close(result.masks.xyn[0], result.masks.xyn[1]))

    furniture_masks = []
    # with open('fikga.pkl', 'wb') as file:
    #     # A new file will be created
    #     pickle.dump(results, file)

    # with open('fikga.pkl', 'rb') as file:
    #     # Call load method to deserialze
    #     result = pickle.load(file)

    for r in result:
        r = r.cpu()
        cls = r.boxes.cls.numpy()[0]
        result = display_mask_on_image(image, r.masks.data.numpy())

        if cls in interested_class:
            for i, points in enumerate(r.masks.xy):
                print("mask")
                cv2.imwrite(f"dioca_{i}.jpg", draw_points_with_lines(frame=image, points=points))

            break
            # Convert float coordinates to integers
            points = points.astype(np.int32)

            # Create a mask with the same size as the input image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # Draw the polygon on the mask
            cv2.fillPoly(mask, [points], 255)

            # Create a colored overlay (you can change the color)
            overlay = image.copy()
            overlay[mask == 255] = [0, 0, 255]  # Red color (BGR format)

            # Blend the original image with the overlay
            alpha = 0.5  # Transparency factor
            result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

            # Add contour around the mask (optional)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            # Then find contours on the dilated mask
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # Green contour

            # Display the result
            cv2.imshow('Result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save the result (optional)
            cv2.imwrite('result.jpg', result)
