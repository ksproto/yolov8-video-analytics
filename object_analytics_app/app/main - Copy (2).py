import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
from flask import Flask, render_template, request, Response
import os

app = Flask(__name__)
model = YOLO('yolov8s.pt')

# Load class names
with open("coco.txt", "r") as f:
    class_list = f.read().splitlines()

tracker = Tracker()
object_counts = {}  # To store total object counts
region_counts = {}  # To store counts per region (will need region definitions)
speed_info = {}    # To store speed information (will require implementation)

# Define regions of interest (example - adjust as needed)
REGIONS = [
    ("Region 1", (100, 100, 300, 200)),  # (name, (x1, y1, x2, y2))
    ("Region 2", (400, 150, 600, 250)),
]

def process_frame(frame):
    results = model.predict(frame)
    detections = results[0].boxes.data.cpu().numpy()
    tracked_objects = tracker.update(detections[:, :4])

    annotated_frame = frame.copy()
    current_frame_detections = []

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = map(int, det)
        class_name = class_list[cls_id]
        if 'car' in class_name:
            current_frame_detections.append([x1, y1, x2, y2])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    tracked_ids = tracker.update(current_frame_detections)

    for track in tracked_ids:
        x1, y1, x2, y2, obj_id = map(int, track)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(annotated_frame, str(obj_id), (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # --- Region Counting (Basic Implementation) ---
    current_region_counts = {region_name: 0 for region_name, _ in REGIONS}
    for region_name, (rx1, ry1, rx2, ry2) in REGIONS:
        cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        for det in detections:
            cx = int((det[0] + det[2]) / 2)
            cy = int((det[1] + det[3]) / 2)
            if rx1 < cx < rx2 and ry1 < cy < ry2:
                current_region_counts[region_name] += 1
        cv2.putText(annotated_frame, f'{region_name}: {current_region_counts[region_name]}', (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    global region_counts
    region_counts = current_region_counts

    global object_counts
    object_counts['total'] = len(tracked_ids)

    # --- Speed Estimation (Needs more sophisticated logic) ---
    # This would involve tracking object positions over time and potentially knowing real-world distances.
    # For now, we'll leave this as a placeholder.
    global speed_info
    speed_info['average'] = "Not implemented"

    return annotated_frame

def generate_frames():
    if 'video_path' in globals() and globals()['video_path']:
        cap = cv2.VideoCapture(globals()['video_path'])
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1020, 500))
            processed_frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
    else:
        # Return a placeholder frame or message if no video is uploaded
        placeholder_frame = np.zeros((500, 1020, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "No video uploaded", (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', placeholder_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path  # Declare video_path as global at the start
    if 'video' not in request.files:
        return "No video file uploaded.", 400
    video_file = request.files['video']
    if video_file.filename == '':
        return "No video file selected.", 400
    if video_file:
        filename = "uploaded_video.mp4"  # You can generate a unique filename
        video_path = os.path.join(app.root_path, 'static', filename)
        video_file.save(video_path)
        globals()['video_path'] = video_path
        return render_template('index.html', upload_success=True)
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    return render_template('logs.html', object_counts=object_counts, region_counts=region_counts, speed_info=speed_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')