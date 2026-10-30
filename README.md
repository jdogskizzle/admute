# admute

Mutes ads on live TV from the HDHomeRun. Jellyfin uses it as an M3U tuner.

- Idle: nothing runs except a tiny web server (no tuner, no GPU).
- When a channel is opened: pulls it from the HDHomeRun, delays it by `DELAY` seconds,
  and runs a GPU-decoding ffmpeg so the detector can look ahead and mute the ad from its start.
- Muting swaps the AC-3 audio frames for silent ones. No re-encoding; video is untouched.
- `LINGER` seconds after the last viewer leaves, the tuner and GPU are released.

## Setup
All machine-specific settings are in `docker-compose.yaml`: HDHomeRun address, channels,
folders, user ID, the Docker network Jellyfin is on, and the GPU. Then `docker compose up -d --build`.

## Setup in Jellyfin
Dashboard > Live TV > Tuner Devices > Add > **M3U Tuner**, URL `http://admute:5004/lineup.m3u`.
Channel IDs match the hdhr-guide XMLTV guide, so the guide lines up automatically.

## How detection works
The station logo. The detector learns where each channel's logo is (edges that stay put in one
corner over many minutes), which takes about 5 minutes of watching a channel the first time.
After that it's saved in `data/logos/<channel>.npz` and reused. An ad is "logo gone for
`AD_MIN` seconds"; it ends when the logo is back for `SHOW_MIN` seconds. Black frames and
silence nudge the start and end to the actual cut.

## Files
- `data/logos/<ch>.png`: picture of what was learned (bright = the logo mask). Check this first if muting is off.
- `data/logs/<ch>/<time>.csv`: one row per second: logo score, black, silent, state. Use for tuning.
- `data/logs/events.log`: every ad start and end.

## Handy commands
```
docker exec admute curl -s localhost:5004/status          # what's open, logo score, muted or not
docker exec admute curl -s localhost:5004/mute/4.1?seconds=20   # manual test mute
docker logs -f admute
```
(`curl` may not be in the image; use `python3 -c "import urllib.request as u;print(u.urlopen('http://localhost:5004/status').read().decode())"`.)

## Tuning (environment in docker-compose.yaml)
`CHANNELS`, `DELAY`, `LINGER`, `LOGO_ON` / `LOGO_OFF` (fractions of the channel's usual logo score,
default 0.85 / 0.75), `MERGE` (seconds; ads closer than this count as one break), `AD_MIN`, `SHOW_MIN`,
`AD_MAX` (unmute if the "ad" lasts this long, e.g. a show with no logo; default 300),
`ARM` (seconds the logo must be on screen before muting is armed), `LEARN_MIN` (minutes).

## Replaying recordings
`./replay` runs the same detector and muting on a recording, faster than real time. The
detector only looks DELAY seconds ahead, like live, so results match what live TV would do.

```
./replay survey path/to/recording.ts                  # whole recording, ~20x real time
./replay render path/to/recording.ts 1:20:00 10:00    # a clip with muting applied
```
- survey writes to `data/replay/<name>/`: `breaks.txt` (what would be muted), `timeline.csv`
  (per second), `timeline.png` (1 px = 5 s: logo score, ad state, black, silence) and
  `sheets/` (one frame every 10 s, 10 minutes per sheet). It learns the logo first if needed.
- render uses the logo the survey learned and writes the clip to the review folder
  (mounted at `/review` in docker-compose.yaml).
- Recordings must be inside a folder mounted in docker-compose.yaml; `./replay` translates
  host paths to container paths from those mounts.

## Not done yet
- Matching repeated ads by audio fingerprint.
- Starting the stream before the full delay builds up (Jellyfin waits about `DELAY` seconds on tune-in).
