# speech_agent

Real-time **driving commentary** for an autonomous vehicle, in the spirit of Wayve
**LINGO-1/2**. Reads an **RViz screencast**, describes each sampled frame with a
local **gemma4** vision model, and **speaks** the commentary aloud whenever the car
begins a new action — turning, slowing, stopping/yielding, resuming. Video is the
only input.

## Pipeline (LangGraph)

```
encode -> perceive -> gate -> narrate
```

- **encode** — downscale the frame to 256×256 and JPEG-encode it for the VLM.
- **perceive** — gemma4 returns structured JSON (maneuver / cause / utterance / confidence).
- **gate** — suppress redundant frames; emit an utterance only when a NEW action begins.
- **narrate** — speak the gated utterance aloud, non-blocking, off the video loop.
  Two voice engines: **gtts** (Google, online) and **pyttsx3** (offline OS voice);
  `--voice auto` prefers gtts and falls back to pyttsx3 when the network drops.

## Layout

```
src/
  configs/prompt.toml  # system + user prompts — edit without touching code
  bot.py               # model settings (name/temp) + prompt; invoke(image_b64) -> JSON text
  speaker.py           # Speaker: gtts + pyttsx3 voice engines, async playback
  main.py              # video loop + LangGraph (encode -> perceive -> gate -> narrate)
```

## Run

```bash
cd src
uv run python main.py                          # default video (../dataset/recording.mp4)
uv run python main.py --video PATH --every 15  # any screencast, 1 of every 15 frames
uv run python main.py --no-speak               # overlay/print only, no audio
uv run python main.py --voice pyttsx3          # offline OS voice (no network)
```

Flags: `--video`, `--model`, `--every` (process 1 of every N frames),
`--out` (annotated video path; `''` to disable), `--max-frames`, `--no-speak`,
`--voice {auto,gtts,pyttsx3}`.

Requires a local [Ollama](https://ollama.com) serving the `gemma4` vision model
(`ollama pull gemma4:e2b-it-qat`). The **gtts** voice needs network access; the
**pyttsx3** voice works fully offline, and `--voice auto` uses gtts when online and
falls back to pyttsx3 otherwise.
