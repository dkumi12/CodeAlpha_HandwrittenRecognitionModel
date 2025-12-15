FROM python:3.9-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install gdown to handle Google Drive downloads
RUN pip install gdown

# 3. Create model directory
RUN mkdir -p model

# 4. DOWNLOAD THE MODEL FROM GOOGLE DRIVE
# We use your specific File ID here:
RUN gdown --id 1g3t2Dtkr2QZG6RJ2oWv2IEpnB7csYrS7 -O model/mobilenet_best.h5

# 5. Copy the rest of the code
COPY main.py .
COPY ui.py .
COPY run_app.sh .

# 6. Make script executable
RUN chmod +x run_app.sh

EXPOSE 7860
CMD ["./run_app.sh"]
