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
- **Watch**: the admute channels live, as Jellyfin gets them (muted, `DELAY` behind), with
  the channel and what's on now and next from the guide. The picture is deinterlaced to 60
  frames a second and sent as HLS in two versions, the broadcast's size at up to `WATCH_MBPS`
  (default 20) and 720p at 5 Mb/s; the player (hls.js, or Safari's own) switches between
  them as the connection allows, about 6 seconds behind the newest piece. The browser
  upscales to the screen's real pixels (Lanczos-3 with anti-ringing, then AMD's RCAS
  sharpening from FSR 1, in WebGL 2; "Off" leaves it to the browser), stepping down by itself
  if the device can't keep up. The bar shows size, bitrate, buffer and dropped frames. One
  channel at a time; it stops 20 seconds after the player goes away, freeing the tuner. Keys:
  space, m, f, ↑/↓ for channels.
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
./replay recap path/to/recording.ts                   # the game's key plays, cut into a recap
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
- listen and sounds: see Listening below. recap: see Rapid recap below.
- Recordings must be inside a folder mounted in docker-compose.yaml; `./replay` translates
  host paths to container paths from those mounts.

## Rapid recap
The key plays of a football game so far, cut from its recording while the game is on, so
catching up starts at once. The studio keeps the recap of every game being recorded up to
date (every `RECAP_EVERY` seconds, default 20); `./replay recap` makes one from any
recording, and also writes it to the review folder as two MP4s (all highlights; scores and
turnovers).
- **Which plays**: from ESPN's public play-by-play (`RECAP_FEED`), matched to the recording by
  the teams in its name. Scores, turnovers and missed kicks; also sacks, gains of 20+ yards
  and long returns. Overturned calls go by the final ruling.
- **When**: ESPN logs each play with a wall-clock time. The recording's own clock (its last
  write, minus its length) says where that falls in the file. The broadcast shows the snap
  some seconds later, and how many differs by game (about 22 s on NBC, 16 s on FOX and CBS in
  the games so far). So each play's snap is looked for in the picture (a still wide shot of the
  field, then everything moving at once), and the plays where it's clear set the game's lag
  for the rest. Clips run from `RECAP_PRE` (4) seconds before the snap to the end of the
  play, longer for long gains and scores, with a caption at the start. A clip whose snap
  moves as the lag becomes clearer is cut again.
- Clips are H.264 at the broadcast's size, encoded on the CPU (a few seconds each;
  `RECAP_GPU=1` uses NVENC). They're kept in `data/recap/<name>/`.
- In the Watch tab: **Catch up** under the player when the channel's game has a recap, and a
  list of every game's recap beside the guide. Clips play one after another (upscaled like
  live), plays added meanwhile are picked up, and it goes back to live at the end. ← and →
  move between plays. `RECAP=0` turns the studio's recap maker off.

## Listening (sponsor reads, jingles, promos)
**Live:** during sports (programs the guide calls sports; renders of recordings too), each
channel's sound is transcribed as it comes in: every `SPEECH_STEP` seconds (1) the last
`SPEECH_WINDOW` seconds (10) go through the model, and a sponsor billboard found in them
(the same rule as below) is muted from its first word to its last, picture left alone.
"Brought to you by" or "powered by" is usually said a second or two into the line, so with
viewers `DELAY` seconds behind it is muted from the start. Lines inside an ad break aren't
looked for. The model is loaded when the first channel starts listening and freed a minute
after the last one stops. `SPEECH=0` turns it off. Events log each muted line.

Jingles and network promos are not used live: on the reviewed games the jingle comes just
before breaks the detector already starts on time, and a house promo is followed by more
ads too often (4 of 15) to unmute on.

**Offline**, to find examples and check the rules:

- `listen` transcribes the commentary with faster-whisper (distil-large-v3, in the image, on
  the GPU in 8-bit: about 1 GB of GPU memory, a 3-hour game in about 4 minutes) into
  `speech.json` (each word with its time), and fingerprints the sound into `prints.npz`
  (20 a second: 32 bits from how 33 frequency bands' loudness changes). `--prints` redoes only
  the fingerprints.
- `sounds` finds, in every listened-to recording:
  - **Sponsor lines** (`reads.json`), in the game (not in a break). Billboards and slogans
    are ads and would be muted: the broadcast's own sponsor tags ("The NFL powered by Verizon
    5G", "The NFL on CBS is sponsored by…", "Sunday Night Football is brought to you by…",
    "You're watching the Allstate halftime"), "is the official … partner of" and promo codes.
    The commentators naming a feature's sponsor ("aerial coverage brought to you by…") or a
    brand in a segment's name ("coming up on the Verizon halftime show") are left on.
  - **Repeated sounds** (`data/sounds/sounds.json`): stretches whose fingerprints line up
    with another stretch, in any recording, and sounds shared by break edges (the league's
    jingle, often under the commentary). Each gets a guess from where it falls against the
    breaks: reviewed labels, else the survey's mutes. Two recordings of the same broadcast
    (a doubleheader's second game recorded while the first was on) are not counted as repeats.
- Your answers on the Sounds page are kept in `data/sounds/verdicts.json`.
- Settings: `SPEECH` (0: don't listen live), `SPEECH_MODEL` (model folder, default the one in
  the image), `SPEECH_DEVICE` (`cuda` or `cpu`; the CPU is too slow to catch lines live),
  `SPEECH_WINDOW`, `SPEECH_STEP`.

## Not done yet
- Starting the stream before the full delay builds up (Jellyfin waits about `DELAY` seconds on tune-in).
