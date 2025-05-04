#!/bin/bash
docker build -t object_analytics_app docker/.
docker run -p 5000:5000 object_analytics_app