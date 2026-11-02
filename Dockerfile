# The image model for the scene check (is this the broadcast or an ad?), exported to ONNX
# here so the running image needs only onnxruntime, not PyTorch.
FROM python:3.12-slim AS scene-model
RUN pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu open_clip_torch==2.29.0 onnx==1.17.0 \
      --extra-index-url https://download.pytorch.org/whl/cpu
COPY tools/export_scene_model.py /tmp/
RUN python /tmp/export_scene_model.py /scene.onnx

# Built on Jellyfin's image to reuse its ffmpeg, which has NVIDIA decoding built in.
FROM jellyfin/jellyfin
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-numpy python3-aiohttp python3-pip \
 && pip install --no-cache-dir --break-system-packages onnxruntime==1.20.1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=scene-model /scene.onnx /opt/models/scene.onnx
# Shared sign-in library for the studio (the formlogin build context, see docker-compose.yaml).
COPY --from=formlogin formlogin /opt/formlogin/formlogin
ENV PYTHONPATH=/opt/formlogin
WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/admute.py"]
