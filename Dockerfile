# Built on Jellyfin's image to reuse its ffmpeg, which has NVIDIA decoding built in.
FROM jellyfin/jellyfin
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-numpy python3-aiohttp \
 && rm -rf /var/lib/apt/lists/*
# Shared sign-in library for the studio (the formlogin build context, see docker-compose.yaml).
COPY --from=formlogin formlogin /opt/formlogin/formlogin
ENV PYTHONPATH=/opt/formlogin
WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/admute.py"]
