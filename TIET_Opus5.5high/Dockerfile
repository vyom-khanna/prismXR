FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run.sh ./
COPY code ./code
COPY weights ./weights
# rejoin the <100 MB weight parts once at build time (MD5-checked), not on every container start
RUN PYTHONPATH=/app/code python -m prism.join_weights
RUN chmod +x run.sh
# usage: docker run --gpus all -v /data:/data prism /data/video.mp4 /data/out.json
ENTRYPOINT ["./run.sh"]
