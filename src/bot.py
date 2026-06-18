"""
``Bot`` — loads the gemma4 vision model (local Ollama) and the prompts, and turns
one video frame (base64 JPEG) into a structured JSON reply.

The model is constrained to :data:`RESPONSE_SCHEMA` via Ollama's structured-output
``format``, so ``invoke`` always returns well-formed JSON (maneuver / commentary)
— reliable for the downstream parse + gate.

The model settings (name, temperature, etc.) are plain ``__init__`` defaults.
gemma4's "thinking" mode is disabled (``reasoning=False``) — otherwise it eats the
small ``num_predict`` budget and returns empty content on image calls.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

try:                       
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_toml(path: str) -> dict:
    """Load a TOML file, resolved relative to this file so the CWD doesn't matter."""
    full = path if os.path.isabs(path) else os.path.join(_HERE, path)
    with open(full, "rb") as file:
        return tomllib.load(file)


# The model is forced to emit JSON matching this schema (Ollama structured output),
# so downstream stages never have to cope with prose, code fences, or malformed JSON.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        # "Maneuver Detected" in the prompt; NO_ACTION_STEADY_STATE means cruising
        # steadily with a clear path -> stay silent (empty commentary).
        "maneuver": {
            "type": "string",
            "enum": [
                "NO_ACTION_STEADY_STATE",
                "SLOWING_DOWN", "STOPPING", "YIELDING",
                "TURNING_LEFT", "TURNING_RIGHT",
                "CHANGING_LANES", "HAZARD_RESPONSE",
                
            ],
        },
        # "Commentary" in the prompt: the first-person passenger-facing speech;
        # empty string when the maneuver is NO_ACTION_STEADY_STATE.
        "commentary": {"type": "string"},
    },
    "required": ["maneuver", "commentary"],
}


class Bot:
    """Loads gemma4 (ChatOllama) + prompts; ``invoke(image_b64)`` returns JSON text."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        temperature: float = 0.2,             # low -> stable, deterministic captions
        num_predict: int = 256,               # small budget; keep replies short
        prompts_path: str = "configs/prompt.toml",
    ):
        prompts = _load_toml(prompts_path)

        self.model_name = model_name or "gemma4:e2b-it-qat"
        self.temperature = temperature
        self.num_predict = num_predict

        self.system_prompt: str = prompts["system"].strip()
        self.user_prompt: str = prompts["user"].strip()

        self.history = []  # recent maneuvers, fed back to the model as memory

        self.model = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
            format=RESPONSE_SCHEMA,  # force structured JSON output
            reasoning=False,  # see module docstring
        )

    def invoke(self, image_b64: str) -> str:
        """Run one vision call on a base64 JPEG and return schema-constrained JSON text.

        A short MEMORY of recent maneuvers is fed in so the model fills ``utterance``
        only when a NEW action begins (and stays silent on continuations/cruising).
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": f"{self.user_prompt}\n\n{self._memory_note()}"},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    },
                ]
            ),
        ]
        reply = self.model.invoke(messages)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        self._remember(text)
        return text

    def _memory_note(self) -> str:
        """Summarize recent maneuvers so the model can detect a NEW action."""
        if not self.history:
            return "MEMORY: this is the first frame — the vehicle has no previous maneuver."
        recent = ", ".join(self.history[-5:])
        return (
            f'MEMORY: the vehicle\'s previous maneuver was {self.history[-1]} '
            f'(recent, oldest->newest: {recent}). Fill "commentary" ONLY if this frame '
            'begins a NEW action different from that previous maneuver; otherwise set '
            '"commentary" to an empty string and the maneuver to NO_ACTION_STEADY_STATE.'
        )

    def _remember(self, reply_text: str) -> None:
        """Record this frame's maneuver so the next call knows the prior state."""
        try:
            maneuver = str(json.loads(reply_text).get("maneuver", "")).strip().upper()
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
            maneuver = ""
        if maneuver:
            self.history.append(maneuver)
            del self.history[:-10]  