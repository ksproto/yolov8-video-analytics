FROM python:3.9-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libgtk-3-dev

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY object_analytics_app/requirements.txt .
COPY object_analytics_app/app /app/app
COPY object_analytics_app/models /app/models
COPY object_analytics_app/models/yolov8s.pt /app/models/yolov8s.pt
COPY object_analytics_app/app/coco.txt /app/
COPY object_analytics_app/app/tracker.py /app/app/


ENV FLASK_APP=app/main.py
EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
