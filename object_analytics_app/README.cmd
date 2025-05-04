# Real-time Object Detection and Tracking Web Application

This project implements a Flask web application for real-time object detection, tracking, region-based counting, and speed estimation using YOLOv8. It supports uploading video files and streaming from a camera.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the Application (Docker)](#running-the-application-docker)
    - [Running the Application (Without Docker)](#running-the-application-without-docker)
- [Folder Structure](#folder-structure)
- [YOLOv8 Model](#yolov8-model)
- [Usage](#usage)
- [Logs and Statistics](#logs-and-statistics)
- [Sample Videos](#sample-videos)
- [Learning Outcomes](#learning-outcomes)
- [Further Improvements](#further-improvements)
- [License](#license)
- [Contact](#contact)

## Overview

This application provides a user-friendly web interface to process video streams for object detection and analysis. It leverages the power of YOLOv8 for real-time object detection and integrates tracking algorithms to maintain object identities across frames. Additionally, it implements region-based counting and speed estimation functionalities for more advanced video analytics.

## Features

- **Video Input:** Supports uploading local video files and streaming from a connected camera.
- **Real-time Object Detection:** Utilizes a trained YOLOv8 model to detect objects in each frame of the video.
- **Object Tracking:** Employs a tracking algorithm (e.g., DeepSORT) to assign unique IDs to detected objects and track their movement across frames.
- **Live Annotated Video:** Displays the video stream with bounding boxes around detected objects and their tracking IDs.
- **Object Counting:**
    - **Total Count:** Shows the total number of unique objects detected in the video.
    - **Region-based Count:** Allows defining specific regions of interest in the video frame and counts the number of objects entering or present in those regions.
- **Speed Estimation:** Estimates the speed of detected objects based on their movement between consecutive frames (requires camera calibration or assumptions about object size).
- **Logs and Statistics:**
    - Displays logs of detected objects and their activities.
    - Provides statistics such as object count per minute/hour.
- **User Interface:** Intuitive web interface built with Flask, HTML, and CSS (optional JavaScript).
- **Dockerized Deployment:** The entire application is containerized using Docker for easy deployment and portability.

## Getting Started

### Prerequisites

- Docker (for containerized deployment)
- Python 3.7+
- pip (Python package installer)
- OpenCV (`opencv-python`)
- PyTorch (`torch`)
- Ultralytics (`ultralytics`)
- Other libraries as listed in `requirements.txt`

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your_github_repo_url>
   cd <your_repo_name>