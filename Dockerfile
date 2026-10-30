# Built on Jellyfin's image to reuse its ffmpeg, which has NVIDIA decoding built in.
FROM jellyfin/jellyfin
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-numpy python3-aiohttp \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/admute.py"]
