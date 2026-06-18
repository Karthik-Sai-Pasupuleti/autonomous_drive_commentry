"""
Driving-commentary pipeline (Wayve LINGO style): play the RViz screencast in a
window, ask the vision model to explain the scene every N frames, overlay its reply
on the video, and *speak* the commentary aloud whenever the vehicle begins a new
action (turning, slowing, stopping/yielding, resuming).

The per-frame LangGraph is ``encode -> perceive -> gate -> narrate``:
encode the frame, perceive it with the VLM, gate out redundant frames so only NEW
actions are announced, then narrate the gated line through the :class:`Speaker`.

Run from ``src/``::

    uv run python main.py                          # default video
    uv run python main.py --video PATH --every 15
    uv run python main.py --no-speak               # overlay/print only, no audio

Press ``q`` or ``Esc`` (or close the window) to quit.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import textwrap
from typing import Any, TypedDict

import cv2
from langgraph.graph import END, START, StateGraph

from bot import Bot
from speaker import Speaker

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.normpath(os.path.join(_HERE, "..", "dataset", "screencast_third_person.webm"))
DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, "..", "output.mp4"))
WINDOW = "speech_agent"
INPUT_SIZE = 256  # the whole pipeline runs at this square size to cut computation


def frame_to_jpeg_b64(image, size: int = INPUT_SIZE, quality: int = 85) -> str:
    """Resize a BGR frame to ``size`` x ``size`` and encode it as a base64 JPEG string.

    A small fixed square keeps the vision model's input cheap and uniform to process.
    """
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def draw_reasoning(frame, text: str, t_sec: float):
    """Return a copy of *frame* with the timestamp + wrapped *text* in a bottom panel."""
    img = frame.copy()
    h, w = img.shape[:2]
    max_chars = max(20, w // 12)
    lines = []
    for para in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(para, max_chars) or [""])

    line_h, pad = 24, 12
    panel_h = pad * 2 + line_h * (len(lines) + 1)  # +1 for the timestamp line
    y0 = max(0, h - panel_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    y = y0 + pad + 14
    cv2.putText(img, f"t = {t_sec:6.2f}s", (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 255), 1, cv2.LINE_AA)
    for line in lines:
        y += line_h
        cv2.putText(img, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class FrameState(TypedDict, total=False):
    """State threaded through the per-frame LangGraph (encode -> perceive -> gate -> narrate)."""

    frame: Any        # source BGR frame (numpy array)
    image_b64: str    # model-ready JPEG
    reasoning: str    # the model's raw reply (perception)
    maneuver: str     # parsed maneuver
    speech: str       # gated announcement (non-empty only when an action begins)
    spoken: bool      # whether this frame produced speech


# Closed set of maneuvers the prompt may return (mirrors RESPONSE_SCHEMA); anything
# else -> NO_ACTION_STEADY_STATE (idle). Every maneuver except the idle one counts
# as "an action" worth announcing.
IDLE_MANEUVER = "NO_ACTION_STEADY_STATE"
KNOWN_MANEUVERS = {
    IDLE_MANEUVER,
    "SLOWING_DOWN", "STOPPING", "YIELDING",
    "TURNING_LEFT", "TURNING_RIGHT",
    "CHANGING_LANES", "HAZARD_RESPONSE",
}


def parse_perception(text: str):
    """Extract ``(maneuver, commentary)`` from the model's JSON reply; tolerant of junk."""
    maneuver, commentary = IDLE_MANEUVER, ""
    s = (text or "").strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(s[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict):
            maneuver = str(data.get("maneuver", "")).strip().upper()
            commentary = str(data.get("commentary", "")).strip()
    if maneuver not in KNOWN_MANEUVERS:
        maneuver = IDLE_MANEUVER
    return maneuver, commentary


def build_graph(bot: Bot, speaker: "Speaker | None" = None):
    """Compile the LangGraph ``encode -> perceive -> gate -> narrate``.

    The ``gate`` layer sits on top of perception and removes redundant frames: it
    emits speech only when the car *begins* an action (any non-idle maneuver),
    then stays silent until the car returns to NO_ACTION_STEADY_STATE. The
    ``narrate`` layer voices that gated speech aloud (LINGO-style commentary)
    without blocking the video loop; with no ``speaker`` it is an inert pass-through.
    """
    memory = {"prev_active": False}  # persists across per-frame invocations

    def encode(state: FrameState) -> dict:
        return {"image_b64": frame_to_jpeg_b64(state["frame"])}

    def perceive(state: FrameState) -> dict:
        try:
            text = bot.invoke(state["image_b64"])
        except Exception as exc:  # one bad frame shouldn't stop the run
            text = f"[error: {exc}]"
        return {"reasoning": text}

    def gate(state: FrameState) -> dict:
        maneuver, commentary = parse_perception(state.get("reasoning", ""))
        active = maneuver != IDLE_MANEUVER
        begins = active and not memory["prev_active"]  # backstop against repeats
        memory["prev_active"] = active
        # The model (with its MEMORY) already emits commentary only on a new
        # action, so trust it; the begin-edge just guards against a stray repeat.
        speech = commentary if (commentary and begins) else ""
        return {"maneuver": maneuver, "speech": speech, "spoken": bool(speech)}

    def narrate(state: FrameState) -> dict:
        # Speak the gated line aloud; say() is non-blocking (off-thread synthesis).
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain an RViz screencast frame by frame.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="path to the screencast")
    parser.add_argument("--model", default=None, help="override the Ollama model tag")
    parser.add_argument("--every", type=int, default=5, help="run the model on 1 of every N frames")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="save the annotated video here (pass '' to disable saving)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = whole video)")
    parser.add_argument("--no-speak", action="store_true",
                        help="disable spoken commentary (text/overlay only)")
    parser.add_argument("--voice", choices=("auto", "gtts", "pyttsx3"), default="auto",
                        help="TTS engine: gtts (Google, online), pyttsx3 (offline OS voice), "
                             "or auto (gtts with offline fallback)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"error: video not found: {args.video}")
        return 1

    bot = Bot(model_name=args.model)
    speaker = Speaker(enabled=not args.no_speak, engine=args.voice)
    graph = build_graph(bot, speaker)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"error: could not open video: {args.video}")
        return 1

    # Some containers (e.g. this webm) report a nonsense FPS like 1000; clamp to a
    # sane range so timestamps and playback speed stay realistic.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 <= fps <= 120.0):
        fps = 30.0
    delay = max(1, int(1000.0 / fps))
    every = max(1, args.every)

    # Windows and macOS use OpenCV's native GUI backend, which needs no $DISPLAY.
    # Only Linux relies on X11: there OpenCV's QT/xcb backend hard-aborts the
    # process (not a catchable exception) when it can't connect, so gate on
    # $DISPLAY up front and fall back to console-only rather than crash.
    show = sys.platform in ("win32", "darwin") or bool(os.environ.get("DISPLAY"))
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    else:
        print("[no $DISPLAY found; running console-only. On Wayland, ensure XWayland "
              "is active, or run from an X session.]")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save = False  # don't write the annotated video; play/process only
    writer = None  # created lazily, once we know the frame size

    caption = ""  # last gated announcement; only changes when an action begins
    max_frames = max(0, args.max_frames)
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Downscale the source frame once, up front: every downstream stage
            # (overlay, model encode, video writer) then works on a small 256x256
            # image instead of the full 2560x1440, cutting decode/copy cost.
            frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            idx += 1
            if max_frames and idx >= max_frames:
                break
            t_sec = idx / fps

            # The overlaid frame is exactly what we both show and save.
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

            # Show the frame first so the window appears immediately (the first
            # model call can take many seconds while gemma4 loads).
            if show:
                cv2.imshow(WINDOW, vis)
                if (cv2.waitKey(delay) & 0xFF) in (ord("q"), 27):  # q / Esc
                    break

            if idx % every == 0:
                result = graph.invoke({"frame": frame})
                if result["spoken"]:  # an action just began -> announce + speak once
                    caption = result["speech"]
                    print(f"[{t_sec:7.2f}s] {result['maneuver']}: {caption}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
            print(f"saved annotated video -> {args.out}")
        speaker.close()  # drain any commentary still being spoken
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
