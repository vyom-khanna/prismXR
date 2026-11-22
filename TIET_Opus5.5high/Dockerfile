FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir numpy pyyaml open_clip_torch
COPY run.sh ./
COPY code ./code
COPY weights ./weights
RUN chmod +x run.sh
# usage: docker run --gpus all -v /data:/data prism ./run.sh /data/video.mp4 /data/out.json
ENTRYPOINT ["./run.sh"]
