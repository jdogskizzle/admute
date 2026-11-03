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
# Speech to text (faster-whisper on the GPU) for finding sponsor reads in the commentary.
# --no-deps keeps the onnxruntime above; numpy stays at Debian's version. The CUDA libraries
# come as wheels.
RUN pip install --no-cache-dir --break-system-packages --no-deps faster-whisper==1.1.1 \
 && echo "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')" > /tmp/keep.txt \
 && pip install --no-cache-dir --break-system-packages --ignore-installed -c /tmp/keep.txt ctranslate2==4.8.5 tokenizers==0.23.3 \
      huggingface-hub==2.4.0 av==19.0.2 requests nvidia-cublas-cu12==12.9.3.13 nvidia-cudnn-cu12==9.27.0.42 \
 && python3 -c "from faster_whisper import download_model; download_model('distil-large-v3', output_dir='/opt/models/speech')"
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.13/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.13/dist-packages/nvidia/cudnn/lib
# Shared sign-in library for the studio (the formlogin build context, see docker-compose.yaml).
COPY --from=formlogin formlogin /opt/formlogin/formlogin
ENV PYTHONPATH=/opt/formlogin
WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/admute.py"]
