import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time
from math import dist

# Load YOLO model
model = YOLO('yolov8s.pt')

# Initialize tracker
tracker = Tracker()

# Define counting lines and offset
cy1 = 322
cy2 = 368
offset = 6

# Initialize dictionaries and lists for speed and counting
vh_down = {}
counter = []
vh_up = {}
counter1 = []

# Capture video
cap = cv2.VideoCapture('D:/object_analytics_app/sample_videos/veh2.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Load class list
my_file = open("D:/object_analytics_app/app/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

     # Object detection
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list_of_bboxes = []  # Store only bounding box coordinates for the tracker

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        confidence = float(row[4])
        d = int(row[5])
        class_name = class_list[d]
        if 'car' in class_name or 'truck' in class_name or 'person' in class_name or 'bicycle' in class_name or 'traffic light' in class_name:
            list_of_bboxes.append([x1, y1, x2, y2]) # Only append bounding box coordinates

    # Object tracking (update tracker with bounding boxes)
    tracked_objects = tracker.update(list_of_bboxes)

    # Speed estimation and counting logic
    for tracked_item in tracked_objects:
        x3, y3, x4, y4, id = tracked_item
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        class_id = -1  # Initialize class ID

        # Find the corresponding detection to get the class label
        for index, row in px.iterrows():
            bx1 = int(row[0])
            by1 = int(row[1])
            bx2 = int(row[2])
            by2 = int(row[3])
            conf = float(row[4])
            det_cls_id = int(row[5])
            det_class_name = class_list[det_cls_id]

            if bx1 <= x3 <= bx2 and by1 <= y3 <= by2 and bx1 <= x4 <= bx2 and by1 <= y4 <= by2:
                class_id = det_cls_id
                break

        if class_id != -1:
            class_name = class_list[class_id]
            # Draw bounding box and class label
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(frame, f'{class_name}-{id}', (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Downward counting and speed estimation
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                vh_down[id] = time.time()
            if id in vh_down:
                if cy2 < (cy + offset) and cy2 > (cy - offset):
                    elapsed_time = time.time() - vh_down[id]
                    if counter.count(id) == 0:
                        counter.append(id)
                        distance = 10  # meters
                        a_speed_ms = distance / elapsed_time
                        a_speed_kh = a_speed_ms * 3.6
                        cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # Upward counting
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                vh_up[id] = time.time()
            if id in vh_up:
                if cy1 < (cy + offset) and cy1 > (cy - offset):
                    elapsed1_time = time.time() - vh_up[id]
                    if counter1.count(id) == 0:
                        counter1.append(id)
                        distance1 = 10  # meters
                        a_speed_ms1 = distance1 / elapsed1_time
                        a_speed_kh1 = a_speed_ms1 * 3.6
                        cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4 - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Draw counting lines and counts
    cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
    cv2.putText(frame, ('L1'), (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
    cv2.putText(frame, ('L2'), (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    d = (len(counter))
    u = (len(counter1))
    cv2.putText(frame, ('goingdown:-') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, ('goingup:-') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Combined Output", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
my_file.close()