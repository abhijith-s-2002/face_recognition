import random
import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime
import time

# opening the file in read mode
my_file = open("utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture(0)  # Initialize webcam (0 is usually the default webcam)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Create directory to save images if not exists
output_dir_images = r"C:\Users\ASUS\OneDrive\Desktop\images"
if not os.path.exists(output_dir_images):
    os.makedirs(output_dir_images)

# Create directory to save video if not exists
output_dir_video = r"C:\Users\ASUS\OneDrive\Desktop\video"
if not os.path.exists(output_dir_video):
    os.makedirs(output_dir_video)

# Define the codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Generate new file name for each run
video_filename = datetime.datetime.now().strftime("recorded_video_%Y-%m-%d_%H-%M-%S.avi")
output_video = cv2.VideoWriter(os.path.join(output_dir_video, video_filename), fourcc, 20.0, (frame_wid, frame_hyt))

# Initialize variables for tracking person detection
person_detected = False

# Set the delay between each image capture (in seconds)
capture_delay = 1  # 1 second delay between captures

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            if class_list[int(clsID)] == "person":
                if not person_detected:
                    person_detected = True
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    cv2.imwrite(os.path.join(output_dir_images, f"detected_image_{timestamp}.jpg"), frame)
                    time.sleep(capture_delay)  # Add delay between captures

                # Draw rectangle around the detected object
                x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), detection_colors[int(clsID)], 2)

                # Draw class name
                cv2.putText(frame, class_list[int(clsID)], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_colors[int(clsID)], 2)

                break
        else:
            person_detected = False

    # Write frame to the video file
    output_video.write(frame)

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoWriter object
output_video.release()

# Release the capture
cap.release()
cv2.destroyAllWindows()