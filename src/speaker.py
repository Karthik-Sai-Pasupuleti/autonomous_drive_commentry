"""
``Speaker`` — turns the agent's gated utterances into *spoken* driving commentary
(Wayve LINGO style).

Two interchangeable voice engines:

* **gtts** — Google Translate TTS. Natural voice, but needs network access; each
  line is synthesized to MP3 and played back.
* **pyttsx3** — the local OS voice (SAPI5 on Windows, NSSpeechSynthesizer on macOS,
  espeak on Linux). Robotic, but works fully offline and needs no playback step.

The ``engine="auto"`` mode prefers **gtts** and automatically falls back to
**pyttsx3** for any line that fails (e.g. the network drops mid-run). Everything
runs on a background worker thread, so the video loop is never blocked, and the
whole thing degrades to printed commentary if no engine is usable.

Playback for the gtts path uses only the standard library — Windows' built-in MCI
(via ``ctypes``), or ``afplay`` / ``ffplay`` / ``mpg123`` on macOS / Linux.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Optional


def _play_windows(path: str) -> None:
    """Play an MP3 synchronously via the Windows Media Control Interface."""
    import ctypes

    mci = ctypes.windll.winmm.mciSendStringW
    alias = "speech_agent_tts"
    # ``mpegvideo`` is the MCI device that handles MP3; ``wait`` blocks until done.
    if mci(f'open "{path}" type mpegvideo alias {alias}', None, 0, None) != 0:
        # Fall back to letting MCI infer the device from the extension.
        mci(f'open "{path}" alias {alias}', None, 0, None)
    mci(f"play {alias} wait", None, 0, None)
    mci(f"close {alias}", None, 0, None)


def _play_subprocess(path: str) -> None:
    """Play *path* with the first available CLI player."""
    for player, flags in (("afplay", []), ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
                          ("mpg123", ["-q"]), ("aplay", [])):
        if shutil.which(player):
            subprocess.run([player, *flags, path], check=False)
            return
    raise RuntimeError("no CLI audio player found")


def _make_player():
    """Return a ``play(path)`` callable for this OS, or ``None`` if none works."""
    if sys.platform == "win32":
        return _play_windows
    # On macOS afplay is always present; elsewhere we need a CLI player installed.
    if sys.platform == "darwin" or any(shutil.which(p) for p in ("ffplay", "mpg123", "aplay")):
        return _play_subprocess
    return None


class Speaker:
    """Speaks queued utterances aloud on a background thread; non-blocking.

    Parameters
    ----------
    enabled : bool
        Master switch. When False every :meth:`say` is a no-op.
    engine : {"auto", "gtts", "pyttsx3"}
        Which voice to use. ``"auto"`` prefers gtts and falls back to pyttsx3
        per-line on failure.
    lang, slow :
        Passed to gTTS (ignored by pyttsx3, which uses the OS default voice).
    """

    def __init__(self, enabled: bool = True, *, engine: str = "auto",
                 lang: str = "en", slow: bool = False):
        self.engine_mode = engine
        self.lang = lang
        self.slow = slow

        self._gtts = None                  # gTTS class, if importable
        self._play = None                  # mp3 player callable, if available
        self._pyttsx3 = None               # pyttsx3 module, if importable
        self._pyttsx3_engine = None        # lazily created inside the worker thread
        self._tmpdir: Optional[str] = None
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

        # Resolve which engines are actually usable, then the run order.
        self._gtts_ok = self._probe_gtts() if engine in ("auto", "gtts") else False
        self._pyttsx3_ok = self._probe_pyttsx3() if engine in ("auto", "pyttsx3") else False
        self._order = self._resolve_order(engine)

        self.enabled = enabled and bool(self._order)
        if self.enabled:
            self._tmpdir = tempfile.mkdtemp(prefix="speech_agent_tts_")
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            print(f"[speech: engine order = {', '.join(self._order)}]")
        elif enabled:
            print("[speech disabled: no usable TTS engine; printing commentary only]")

    # -- backend probing -----------------------------------------------------
    def _probe_gtts(self) -> bool:
        try:
            from gtts import gTTS
        except Exception as exc:
            print(f"[gtts unavailable: {exc}]")
            return False
        self._gtts = gTTS
        self._play = _make_player()
        if self._play is None:
            print("[gtts unavailable: no audio player for MP3 playback]")
            return False
        return True

    def _probe_pyttsx3(self) -> bool:
        try:
            import pyttsx3
        except Exception as exc:
            print(f"[pyttsx3 unavailable: {exc}]")
            return False
        self._pyttsx3 = pyttsx3
        return True

    def _resolve_order(self, engine: str):
        """Build the ordered list of engines to try per line."""
        if engine == "gtts":
            return ["gtts"] if self._gtts_ok else []
        if engine == "pyttsx3":
            return ["pyttsx3"] if self._pyttsx3_ok else []
        # auto: prefer gtts, fall back to pyttsx3
        order = []
        if self._gtts_ok:
            order.append("gtts")
        if self._pyttsx3_ok:
            order.append("pyttsx3")
        return order

    # -- public API ----------------------------------------------------------
    def say(self, text: str) -> None:
        """Queue *text* to be spoken. Returns immediately (synthesis is off-thread)."""
        if self.enabled and text:
            self._queue.put(text)

    def close(self) -> None:
        """Wait for queued lines to finish speaking, then stop the worker."""
        if not self.enabled:
            return
        self._queue.put(None)  # sentinel
        if self._thread is not None:
            self._thread.join(timeout=20)
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # -- worker --------------------------------------------------------------
    def _worker(self) -> None:
        counter = 0
        while True:
            text = self._queue.get()
            try:
                if text is None:  # sentinel -> shut down
                    return
                for name in self._order:  # try engines in order; stop on first success
                    try:
                        if name == "gtts":
                            self._speak_gtts(text, counter)
                        else:
                            self._speak_pyttsx3(text)
                        break
                    except Exception as exc:
                        print(f"[{name} failed: {exc}]")
                counter += 1
            finally:
                self._queue.task_done()

    def _speak_gtts(self, text: str, counter: int) -> None:
        path = os.path.join(self._tmpdir or "", f"line_{counter}.mp3")
        self._gtts(text=text, lang=self.lang, slow=self.slow).save(path)
        self._play(path)  # type: ignore[misc]  # blocks until the clip ends

    def _speak_pyttsx3(self, text: str) -> None:
        # The engine must live on the thread that drives it; create it lazily here.
        if self._pyttsx3_engine is None:
            self._pyttsx3_engine = self._pyttsx3.init()  # type: ignore[union-attr]
        engine = self._pyttsx3_engine
        try:
            engine.say(text)
            engine.runAndWait()  # blocks until the line is spoken
        except RuntimeError:
            # A stuck run loop: rebuild the engine once and retry.
            self._pyttsx3_engine = self._pyttsx3.init()  # type: ignore[union-attr]
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
