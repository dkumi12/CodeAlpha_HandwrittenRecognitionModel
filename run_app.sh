#!/bin/bash

# 1. Start FastAPI in the background on port 8000
# The '&' symbol pushes it to the background
uvicorn main:app --host 0.0.0.0 --port 8000 &

# 2. Wait a few seconds for the API to start
sleep 5

# 3. Start Streamlit on port 7860 (Hugging Face's default port)
streamlit run ui.py --server.port 7860 --server.address 0.0.0.0
