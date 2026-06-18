"""
Driving-commentary pipeline (Wayve LINGO style): play the RViz screencast in a
window, ask the vision model to explain the scene every N frames, overlay its reply
on the video, and *speak* the commentary aloud whenever the vehicle begins a new
action (turning, slowing, stopping/yielding, resuming).

The per-frame LangGraph is `encode -> perceive -> gate -> narrate`:
encode the frame, perceive it with the VLM, gate out redundant frames so only NEW
actions are announced, then narrate the gated line through the `Speaker`.
"""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import deque
from pathlib import Path
from typing import Any, TypedDict

import cv2
from langgraph.graph import END, START, StateGraph
from typing_extensions import Required

from bot import Bot, IDLE_MANEUVER, MANEUVERS
from speaker import Speaker

_HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = str((_HERE / ".." / "dataset" / "screencast_third_person_720p.mp4").resolve())
DEFAULT_OUT = str((_HERE / ".." / "output.mp4").resolve())

WINDOW = "speech_agent"
MODEL_LONG_SIDE = 512   # model input: longest edge in px; aspect ratio preserved (no square squash)
TEMPORAL_FRAMES = 1     # frames sent per perception. 1 = single frame.
TEMPORAL_STRIDE = 1     # perception steps between sampled frames.


def frame_to_jpeg_b64(image: Any, long_side: int = MODEL_LONG_SIDE, quality: int = 90) -> str:
    """Downscale a BGR frame so its longer edge is `long_side` px, preserving aspect
    ratio, and encode it as a base64 JPEG string.
    """
    h, w = image.shape[:2]
    scale = long_side / float(max(h, w))
    if scale < 1.0:  # only ever downscale; never upscale a small frame
        image = cv2.resize(
            image, 
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def draw_reasoning(frame: Any, text: str, t_sec: float) -> Any:
    """Return a copy of *frame* with the timestamp + wrapped spoken *text* in a bottom panel."""
    img = frame.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    s = max(0.5, w / 640.0)            # scale factor relative to the 640px baseline
    ts_scale = 0.5 * s                 # timestamp header
    tx_scale = 0.7 * s                 # spoken commentary
    thick = max(1, round(1.6 * s))
    line_h = int(34 * s)
    pad = int(14 * s)

    char_w = max(1, cv2.getTextSize("M", font, tx_scale, thick)[0][0])
    max_chars = max(10, (w - 2 * pad) // char_w)
    
    lines: list[str] = []
    for para in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(para, max_chars) or [""])

    panel_h = pad * 2 + line_h * (len(lines) + 1)  # +1 for the timestamp line
    y0 = max(0, h - panel_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    y = y0 + pad + int(line_h * 0.6)
    cv2.putText(img, f"t = {t_sec:6.2f}s", (pad, y),
                font, ts_scale, (120, 220, 255), thick, cv2.LINE_AA)
    for line in lines:
        y += line_h
        cv2.putText(img, line, (pad, y),
                    font, tx_scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


class FrameState(TypedDict, total=False):
    """State threaded through the per-frame LangGraph (encode -> perceive -> gate -> narrate)."""
    frame: Required[Any]  # source BGR frame (numpy array); always supplied at invoke
    t_sec: float          # timestamp of this frame (for the gate's cooldown)
    images_b64: list[str] # recent model-ready JPEGs, oldest -> newest (motion context)
    maneuver: str         # parsed maneuver directly from Bot
    commentary: str       # reasoning/commentary directly from Bot
    speech: str           # gated announcement (non-empty only when a NEW action begins)
    spoken: bool          # whether this frame produced speech


KNOWN_MANEUVERS = frozenset(MANEUVERS)

MANEUVER_THEME = {
    "SLOWING_DOWN": "slowing", "STOPPING": "slowing", "YIELDING": "slowing",
    "HAZARD_RESPONSE": "slowing",
    "TURNING_LEFT": "turning", "TURNING_RIGHT": "turning",
    "CHANGING_LANES": "lane_change",
}

COOLDOWN_SEC = 10.0  # don't re-announce the same theme within this many seconds of playback
MIN_GAP_SEC = 3.0    # hard floor between ANY two announcements (debounces rapid theme flip-flop)


def build_graph(bot: Bot, speaker: Speaker | None = None) -> Any:
    """Compile the LangGraph `encode -> perceive -> gate -> narrate`."""
    memory: dict[str, Any] = {"last_theme": None, "last_t": -1e9}
    buffer: deque[str] = deque(maxlen=(TEMPORAL_FRAMES - 1) * TEMPORAL_STRIDE + 1)

    def encode(state: FrameState) -> dict[str, Any]:
        buffer.append(frame_to_jpeg_b64(state["frame"]))
        frames = list(buffer)
        last = len(frames) - 1
        idxs = sorted({max(0, last - k * TEMPORAL_STRIDE) for k in range(TEMPORAL_FRAMES)})
        return {"images_b64": [frames[i] for i in idxs]}

    def perceive(state: FrameState) -> dict[str, Any]:
        try:
            images = state.get("images_b64") or []
            # We now unpack the tuple directly from the updated Bot
            maneuver, commentary = bot.invoke(images)
        except Exception as exc:  # one bad frame shouldn't kill the run
            print(f"[perceive error: {exc}]", flush=True)
            maneuver, commentary = IDLE_MANEUVER, ""
            
        return {"maneuver": maneuver, "commentary": commentary}

    def gate(state: FrameState) -> dict[str, Any]:
        maneuver = state.get("maneuver", IDLE_MANEUVER)
        commentary = state.get("commentary", "")
        
        theme = MANEUVER_THEME.get(maneuver)  # None for the idle maneuver
        t = state.get("t_sec", 0.0)
        gap = t - memory["last_t"]
        
        announce = bool(theme and commentary and gap >= MIN_GAP_SEC
                        and (theme != memory["last_theme"] or gap >= COOLDOWN_SEC))
        
        if announce:
            memory["last_theme"] = theme
            memory["last_t"] = t
            
        speech = commentary if announce else ""
        return {"speech": speech, "spoken": bool(speech)}

    def narrate(state: FrameState) -> dict[str, Any]:
        if speaker is not None:
            speaker.say(state.get("speech", ""))
        return {}

    graph = StateGraph(FrameState)
    graph.add_node("encode", encode)
    graph.add_node("perceive", perceive)
    graph.add_node("gate", gate)
    graph.add_node("narrate", narrate)
    graph.add_edge(START, "encode")
    graph.add_edge("encode", "perceive")
    graph.add_edge("perceive", "gate")
    graph.add_edge("gate", "narrate")
    graph.add_edge("narrate", END)
    
    return graph.compile()


def finalize_video(video_path: str, events: list[tuple[float, str]], speaker: Speaker) -> bool:
    """Re-encode *video_path* in place to a widely-playable H.264/AAC MP4."""
    if shutil.which("ffmpeg") is None:
        print("[ffmpeg not on PATH; cannot finalize — file left as MPEG-4, may not play]")
        return False

    tmpdir = tempfile.mkdtemp(prefix="speech_agent_mux_")
    final = f"{video_path}.final.mp4"
    
    venc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart"]
    
    try:
        clips: list[tuple[float, str]] = []  # (t_sec, audio_path)
        for i, (t_sec, text) in enumerate(events):
            # Resolve to standard string paths for OS/subprocess safety
            path = speaker.synthesize(text, str(Path(tmpdir) / f"clip_{i}"))
            if path:
                clips.append((t_sec, path))

        if clips:
            inputs, filters, labels = ["-i", video_path], [], []
            for i, (t_sec, path) in enumerate(clips):
                inputs += ["-i", path]
                ms = max(0, int(round(t_sec * 1000)))
                filters.append(f"[{i + 1}:a]adelay={ms}:all=1[d{i}]")
                labels.append(f"[d{i}]")
                
            if len(clips) == 1:
                filtergraph = filters[0].replace("[d0]", "[aout]")
            else:
                filtergraph = (";".join(filters) + ";" + "".join(labels)
                               + f"amix=inputs={len(clips)}:normalize=0[aout]")
                
            cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filtergraph,
                   "-map", "0:v:0", "-map", "[aout]", *venc, "-c:a", "aac", final]
        else:
            print("[no commentary to add; re-encoding to a silent but playable file]")
            cmd = ["ffmpeg", "-y", "-i", video_path, "-map", "0:v:0", *venc, final]

        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-800:]
            print(f"[ffmpeg finalize failed; original file left as-is]\n{tail}")
            return False
            
        Path(final).replace(video_path)  # atomic swap using pathlib
        return bool(clips)
        
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if Path(final).exists():
            Path(final).unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain an RViz screencast frame by frame.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="path to the screencast")
    parser.add_argument("--model", default=None, help="override the Ollama model tag")
    parser.add_argument("--every", type=int, default=5, help="run the model on 1 of every N frames.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="save the annotated video here (pass '' to disable)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = whole video)")
    parser.add_argument("--no-speak", action="store_true", help="disable spoken commentary (text/overlay only)")
    parser.add_argument("--voice", choices=("auto", "gtts", "pyttsx3"), default="auto", help="TTS engine")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"error: video not found: {args.video}")
        return 1

    bot = Bot(model_name=args.model)
    speaker = Speaker(enabled=not args.no_speak, engine=args.voice)
    graph = build_graph(bot, speaker)
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        cap.release()
        print(f"error: could not open video: {args.video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 <= fps <= 120.0):
        fps = 30.0
    delay = max(1, int(1000.0 / fps))
    every = max(1, args.every)

    show = sys.platform in ("win32", "darwin") or bool(sys.modules.get("os").environ.get("DISPLAY"))
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    else:
        print("[no $DISPLAY found; running console-only.]")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    save = bool(args.out)
    writer = None 

    caption = "" 
    audio_events: list[tuple[float, str]] = [] 
    max_frames = max(0, args.max_frames)
    idx = -1
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
                
            idx += 1
            if max_frames and idx >= max_frames:
                break
            t_sec = idx / fps

            vis = draw_reasoning(frame, caption, t_sec)

            if save:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        print(f"[could not open writer for {args.out}; not saving]")
                        save = False
                        writer = None
                if writer is not None:
                    writer.write(vis)

            if show:
                cv2.imshow(WINDOW, vis)
                if (cv2.waitKey(delay) & 0xFF) in (ord("q"), 27):
                    break

            if idx % every == 0:
                if idx == 0:
                    print("[loading model; the first prediction can take a while ...]", flush=True)
                
                result = graph.invoke({"frame": frame, "t_sec": t_sec})
                maneuver = result.get("maneuver", IDLE_MANEUVER)
                
                if result.get("spoken"):
                    caption = result["speech"]
                    audio_events.append((t_sec, caption))
                    print(f"[{t_sec:7.2f}s] predict={maneuver} -> SPEAK: {caption}", flush=True)
                else:
                    print(f"[{t_sec:7.2f}s] predict={maneuver}", flush=True)
                    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
        speaker.close()

    if audio_events:
        print("\n=== commentary transcript ===")
        for t_sec, text in audio_events:
            print(f"[{t_sec:7.2f}s] {text}")
        print(f"=== {len(audio_events)} line(s) ===\n")
    else:
        print("\n[no commentary was generated for this run]\n")

    if writer is not None and save:
        events = [] if args.no_speak else audio_events
        if events:
            print(f"[finalizing {args.out} with {len(events)} commentary clip(s) ...]")
        had_voice = finalize_video(args.out, events, speaker)
        print(f"saved annotated video {'with voice' if had_voice else '(silent)'} -> {args.out}")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())