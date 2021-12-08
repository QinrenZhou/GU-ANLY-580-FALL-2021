#!/bin/bash

gunicorn query_matching_demo:app \
  --workers 1 \
  --worker-class virtex.VirtexWorker \
  --bind 10.100.2.195:11580 \
  --worker-connections 10000 \
  --timeout 60 \
  --log-level INFO
