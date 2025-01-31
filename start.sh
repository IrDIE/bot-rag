#!/bin/bash
gunicorn main:app --workers 4 -t 90 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 