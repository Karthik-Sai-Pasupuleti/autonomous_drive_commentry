"""
Bot — loads a local Ollama vision model and turns one or more recent video frames
into a structured {maneuver, commentary} JSON reply.

Model choice is a speed/accuracy trade. The default is **gemma4:12b**, a non-thinking VLM:
~0.8s/frame (near real-time over the whole video). Thinking is auto-disabled for gemma.
A qwen3-vl thinking model is more careful on direction but ~6s/frame; pass that as 
`model_name` and `reasoning` is left on automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, cast, get_args

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_HERE = Path(__file__).resolve().parent

# Define the allowed maneuvers as a strict Literal type for Pydantic validation
ManeuverType = Literal[
    "NO_ACTION_STEADY_STATE",
    "SLOWING_DOWN", 
    "STOPPING", 
    "YIELDING",
    "TURNING_LEFT", 
    "TURNING_RIGHT",
    "CHANGING_LANES",
    "HAZARD_RESPONSE"
]

# Re-exported for main.py's gate/parse, derived from the Literal so there is one source of truth.
IDLE_MANEUVER = "NO_ACTION_STEADY_STATE"
MANEUVERS: tuple[str, ...] = get_args(ManeuverType)


class OutputSchema(BaseModel):
    """The structured output schema expected from the Vision Model."""
    maneuver: ManeuverType = Field(
        default="NO_ACTION_STEADY_STATE",
        description="The specific driving action the vehicle should take."
    )
    commentary: str = Field(
        default="",
        description="A brief explanation reasoning about the scene and the chosen maneuver."
    )


def _load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file, resolved relative to this file."""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = _HERE / file_path
        
    with file_path.open("rb") as file:
        return tomllib.load(file)


class Bot:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        temperature: float = 0.0,             
        num_predict: int = 2048,              
        reasoning: bool | None = None,     
        prompts_path: str | Path = "configs/prompt.toml",
    ) -> None:
        prompts = _load_toml(prompts_path)

        self.model_name: str = model_name or "nemotron3:33b"
        self.temperature: float = temperature
        self.num_predict: int = num_predict

        self.system_prompt: str = str(prompts.get("System", "")).strip()
        self.user_prompt: str = str(prompts.get("User", "")).strip()

        if reasoning is None:
            reasoning = False if "gemma" in self.model_name.lower() else None

        # 1. Initialize the base ChatOllama model
        base_model = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
            seed=0,
            reasoning=None,
        )
        
        # 2. Bind the Pydantic schema to force structured JSON output
        self.vision_model = base_model.with_structured_output(OutputSchema)

    def _build_messages(self, images: list[str]) -> list[BaseMessage]:
        """Stage 1 — Formats the prompts and base64 images into a LangChain message payload."""
        if not images:
            raise ValueError("The 'images' list cannot be empty.")

        frame_context = (
            "the current frame" 
            if len(images) == 1 
            else f"{len(images)} consecutive frames, oldest first and newest last"
        )

        # FIX 2: Use list[Any] to satisfy LangChain's strict invariant dict typing
        content: list[Any] = [
            {"type": "text", "text": f"{self.user_prompt}\n\n(You are given {frame_context}.)"}
        ]
        
        content.extend(
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}
            for b64 in images
        )

        # FIX 1: Explicitly type the list as list[BaseMessage] before returning
        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=content)
        ]

        return messages
    
    def invoke(self, frames: str | list[str]) -> tuple[str, str]:
        """Process frames and return the maneuver and commentary."""
        images = [frames] if isinstance(frames, str) else list(frames)
        
        # 1. Build the payload
        messages = self._build_messages(images)
        
        prediction = self.vision_model.invoke(messages)

        print(prediction)
        # 2. Invoke the model (returns the validated Pydantic OutputSchema)
        output = cast(OutputSchema, self.vision_model.invoke(messages))

        
        # 3. Return the fields directly from the Pydantic model
        return output.maneuver, output.commentary