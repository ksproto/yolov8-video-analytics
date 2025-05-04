FROM python:3.9-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libgtk-3-dev

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app /app/app
COPY models /app/models
COPY app/coco.txt /app/
COPY app/tracker.py /app/app/
COPY models/yolov8s.pt /app/models/yolov8s.pt


ENV FLASK_APP=app/main.py
EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
