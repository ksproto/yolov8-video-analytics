
# Intelligent Object Counting and Speed Monitoring System using YOLOv8
Objective 
To develop a real-time system for detecting, tracking, counting, and analyzing object 
movement in surveillance footage using YOLOv8. The system should be capable of 
region-based counting and estimating object speed, and deployed as a Flask-based web app 
containerized with Docker. 

Features 
- Trained YOLOv8 model weights or training instructions.
- Flask web application with UI for video input, visual outputs, and stats. 
- Dockerfile with deployment steps. 
- Sample videos showcasing: 
- Object detection & tracking 
- Region-wise counting 
- Speed estimation 



Key Considerations:

YOLOv8 Integration: The ultralytics library provides a straightforward way to load and run YOLOv8 models in Python.
Object Tracking: Implementing robust object tracking can be complex. Consider using well-established tracking algorithms or libraries.
Region-wise Counting: You'll need a way for users to define these regions, either through hardcoding coordinates or providing a more interactive way through the UI.
Speed Estimation: This is the most challenging part. Accurate speed estimation often requires camera calibration (estimating the camera's intrinsic and extrinsic parameters). A simpler approach might involve assuming a fixed object size and using the change in pixel distance over time, but this will be less accurate.
