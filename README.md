# admute

Mutes ads on live TV. It sits between a tuner and Jellyfin: it reads the tuner's M3U
playlist and serves its own, which Jellyfin uses as an M3U tuner.

Works with any tuner or IPTV source that publishes an M3U playlist of MPEG-TS streams
(HDHomeRun, Tvheadend, Threadfin and others). Muting currently needs AC-3 audio, which is
what US over-the-air broadcasts use; other audio formats pass through unmuted.

- Idle: nothing runs except a tiny web server (no tuner, no GPU).
- When a channel is opened: pulls it from the tuner, delays it by `DELAY` seconds,
  and runs a GPU-decoding ffmpeg so the detector can look ahead and mute the ad from its start.
- Muting swaps the AC-3 audio frames for silent ones. No re-encoding; video is untouched.
- `LINGER` seconds after the last viewer leaves, the tuner and GPU are released.

## Setup
All machine-specific settings are in `docker-compose.yaml`: the tuner's playlist (`SOURCE_M3U`,
a URL or a file; for an HDHomeRun it's `http://<ip>/lineup.m3u`), channels,
folders, user ID, the Docker network Jellyfin is on, and the GPU. Then `docker compose up -d --build`.

## Setup in Jellyfin
Dashboard > Live TV > Tuner Devices > Add > **M3U Tuner**, URL `http://admute:5004/lineup.m3u`.
Guide IDs (`tvg-id`, or `channel-id`, else the channel number) are passed through from the
source playlist, so an XMLTV guide that matches the tuner also matches these channels.

## How detection works
On-screen graphics: the station logo, and in sports the scorebug. The detector learns where
they are (edges that stay put in a corner over many minutes; up to two graphics in different
corners), which takes about 5 minutes of watching a channel the first time. It then learns a
picture of each one and recognises it by shape, which still works for a see-through logo over
a bright background. It's saved in `data/logos/<channel>.npz` and reused.

Sports also have a playing surface (ice, grass) that fills the screen during play, even in
replays that drop the graphics, but rarely in ads. The detector learns its colour, and while it
stays on screen (`SURFACE_HOLD` seconds) a missing logo doesn't start a mute.

Some channels (CBS) cut to every break with a black frame and/or silence, while their own
stat graphics and studio cut-ins have none. The detector learns which channels do that, and on
those it only mutes a lost logo from such a cut, or after `SPLICE_WAIT` seconds.

Side-by-side ads (the game in a small box next to an ad, inside a fixed network frame such as
CBS's "Eye on the field") are recognised by that frame. Tag some as SbS in the studio's labels
and learn them (studio Recordings page, or `./replay layouts`): it learns each frame's steady
edges, checks them against everything else in the labelled recordings so they don't fire on
the game or the network's own boxed score updates, and saves them in `data/layouts/`. Every
channel watches for them, and one on screen mutes right away.

**Scene check (the main signal once trained).** An image model (CLIP, run with onnxruntime
on the CPU, about 35 ms a frame) describes one frame a second, and a small classifier learned
from your reviewed games turns that into the chance it's an ad. It knows a stat card over a
sideline shot, a studio cut-in or the commentators coming back from a break are the
broadcast, even with every logo gone, which the logo rules can't. It mutes when the last
`SCENE_WINDOW` seconds average above `SCENE_ON` and unmutes below `SCENE_OFF`, with the
boundaries placed where the chance crossed one half (as far as `DELAY` allows) and snapped to a
black frame or silence. Train it in the studio (Recordings page) or with `./replay train`
after reviewing games; it reports how it does on each game when that game is left out. On
the first two games (FOX and CBS) that cut ads heard by about 75% and game muted by 15-80%,
with fewer wrong mutes. It's only used for the kinds of program it learned from, by guide
title (e.g. "NFL Football"): live TV checks the guide (`GUIDE_XMLTV`) for the program on now,
recordings use their file name. A football-trained model did badly on hockey, so hockey keeps
the logo rules until a hockey game is reviewed and trained on. Everything else uses the logo
rules below.

It's a show while any graphic is on screen. An ad is "all graphics gone for `AD_MIN` seconds";
it ends when one is back for `SHOW_MIN` seconds. Following two graphics matters for sports:
replays and close-ups usually drop the scorebug but keep the station logo. Black frames and
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
`CHANNELS`, `DELAY`, `LINGER`, `LOGO_ON` / `LOGO_OFF` (shape match of a graphic, 0 to 1, above which
it counts as on screen / below which it's gone; default 0.35 / 0.3), `GRAPHICS` (how many to follow, default 2), `MERGE` (seconds; ads closer than this count as one break), `AD_MIN`, `SHOW_MIN`,
`AD_MAX` (unmute if the "ad" lasts this long, e.g. a show with no logo; default 300),
`SURFACE` (0 turns off the playing-surface check), `SURFACE_AT` (where its threshold sits between
the share usual in ads, 0, and in the show, 1; default 0.5), `SURFACE_HOLD` (seconds; default 2),
`SPLICE` (wait for a black frame or silence before muting: -1 learn per channel, 0 never, 1 always),
`SPLICE_WAIT` (mute anyway after this many seconds; default 20), `SCENE` (0 ignores the scene check), `SCENE_ON` / `SCENE_OFF` (0.9 / 0.5), `SCENE_WINDOW` (seconds, 6),
`LAYOUT` (0 ignores learned
side-by-side layouts), `LAYOUT_HOLD` (seconds a layout must match; default 0.6), `SNAP`,
`ARM` (seconds the logo must be on screen before muting is armed), `LEARN_MIN` (minutes).

## Studio (dashboard)
The `studio` service is a web dashboard for checking and tuning the detector on recordings:

- **Recordings**: every `.ts` in the recordings folder; start a survey of any of them.
- **A recording**: the video (streamed from any point, transcoded on the GPU; as HLS for
  Safari and iPhones, at `api/rec/<name>/hls.m3u8?h=720`) above a zoomable
  timeline of what the detector saw: each graphic's shape match, the playing-surface share and
  its threshold, black frames, silence, and the mutes. Hover for a thumbnail, click to play from
  there. While playing, the sound is muted the way the survey (or tuned settings, or your
  labels) would mute it.
- **Labels**: mark where the ads really are (shift+drag on the timeline, or `I`/`O` while
  watching). Tick "Reviewed" and the recording counts when scoring.
- **Tune**: sliders for the settings. Each change replays the survey's saved trace in about a
  second, without decoding video, and scores it against the labels: seconds of ads heard,
  seconds of game muted, wrong mutes, late starts. "Score all reviewed recordings" checks the
  settings across every game, so tuning one game doesn't break another. It shows the lines to
  put in `docker-compose.yaml` when you find better values.
- **Sounds**: what the commentary says and which sounds come back (see Listening below).
  Sponsor reads found in the game, to mark Mute or Keep, and sounds heard more than once with
  a guess at what they are (jingle at a break's edge, network promo that ends breaks, ad,
  part of the game), to confirm. ▶ plays each one.
- **Live**: what the proxy is doing now, and recent mute events.
- **Jobs**: surveys and 10-minute clip renders started from the dashboard.

It has its own sign-in (the shared [formlogin](https://github.com/jdogskizzle/formlogin)
library): set the password with `docker compose exec studio python3 -m formlogin passwd`
(saved in `data/studio/login.json`, used right away), or on first start on the setup page,
with the code from `docker logs admute-studio`. It
has no host port; put it behind a reverse proxy on the same Docker network, at any path (it
uses relative links), e.g. Caddy `handle_path /admute/* { reverse_proxy admute-studio:8000 }`.

## Replaying recordings
`./replay` runs the same detector and muting on a recording, faster than real time. The
detector only looks DELAY seconds ahead, like live, so results match what live TV would do.

```
./replay survey path/to/recording.ts                  # whole recording, ~20x real time
./replay render path/to/recording.ts 1:20:00 10:00    # a clip with muting applied
./replay layouts                                      # learn side-by-side ad layouts from labels
./replay train                                        # learn the scene check from reviewed games
./replay listen path/to/recording.ts                  # transcribe + fingerprint the sound (GPU)
./replay sounds                                       # sponsor reads and repeated sounds, all recordings
```
- survey writes to `data/replay/<name>/`: `breaks.txt` and `result.json` (what would be muted),
  `trace.npz` (per-frame scores, for tuning in the studio), `timeline.csv` (per second),
  `timeline.png` (1 px = 5 s: logo score, ad state, black, silence) and `sheets/` (one frame
  every 10 s, 10 minutes per sheet). It learns the logo first if needed, and keeps what it
  learned for the next survey of that recording (`--fresh` learns from scratch).
- Recordings decode on the CPU (MPEG-2 is cheap, and parallel surveys would queue on the GPU's
  video decoder); `REPLAY_GPU=1` uses the GPU.
- render uses the logo the survey learned and writes the clip to the review folder
  (mounted at `/review` in docker-compose.yaml).
- listen and sounds: see Listening below.
- Recordings must be inside a folder mounted in docker-compose.yaml; `./replay` translates
  host paths to container paths from those mounts.

## Listening (sponsor reads, jingles, promos)
Not used live yet: this finds examples in recordings to confirm first.

- `listen` transcribes the commentary with faster-whisper (distil-large-v3, in the image, on
  the GPU in 8-bit: about 1 GB of GPU memory, a 3-hour game in about 4 minutes) into
  `speech.json` (each word with its time), and fingerprints the sound into `prints.npz`
  (20 a second: 32 bits from how 33 frequency bands' loudness changes). `--prints` redoes only
  the fingerprints.
- `sounds` finds, in every listened-to recording:
  - **Sponsor reads** (`reads.json`): a line in the game (not in a break) with "brought to you
    by", "presented by", "powered by", "is the official … of" and the like, cut to its clause
    and the brand after it. Lines under 3 s are short mentions, and a brand on a segment's
    name ("the Allstate halftime") is left alone.
  - **Repeated sounds** (`data/sounds/sounds.json`): stretches whose fingerprints line up
    with another stretch, in any recording, and sounds shared by break edges (the league's
    jingle, often under the commentary). Each gets a guess from where it falls against the
    breaks: reviewed labels, else the survey's mutes.
- Your answers on the Sounds page are kept in `data/sounds/verdicts.json`.
- Settings: `SPEECH_MODEL` (model folder, default the one in the image), `SPEECH_DEVICE`
  (`cuda` or `cpu`).

## Not done yet
- Using sponsor reads, jingles and network promos live.
- Starting the stream before the full delay builds up (Jellyfin waits about `DELAY` seconds on tune-in).
