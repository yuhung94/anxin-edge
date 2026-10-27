"""Tier-2 VLM verifier: Qwen2-VL-2B-Instruct looks at ONE keyframe (held in RAM only)
and answers a constrained multiple-choice question about the person's situation.

    cpu   -> Hugging Face Qwen/Qwen2-VL-2B-Instruct on CPU / Apple MPS (Stage I)
    hailo -> Qwen2-VL-2B-Instruct.hef on UGen300 via hailo_platform.genai.VLM

The Hailo .hef takes a 336x336 RGB frame, so the CPU backend is fed the same
336x336 resolution to keep Stage I accuracy representative of Stage II.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

INPUT_SIZE = 336

CHOICES = {
    "A": "standing or walking",
    "B": "sitting on a chair or sofa",
    "C": "lying or sitting on a bed or sofa",
    "D": "bending, kneeling or crouching on purpose (e.g. picking something up)",
    "E": "fallen and lying on the floor",
}
FALL_CHOICES = {"E"}

PROMPT = (
    "You are a home-safety assistant for an elderly person living alone. "
    "Look at the person in this image. Which ONE option best describes them?\n"
    + "\n".join(f"{k}) {v}" for k, v in CHOICES.items())
    + "\nAnswer with the letter first, then one short sentence describing the scene."
)


def parse_answer(text: str) -> tuple[str | None, str]:
    m = re.search(r"\b([A-E])\b", text.strip())
    return (m.group(1) if m else None), text.strip()


def _prep(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)


class HFQwen2VL:
    name = "cpu:Qwen2-VL-2B-Instruct"

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", device: str | None = None):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        px = INPUT_SIZE * INPUT_SIZE
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=px, max_pixels=px)

    def ask(self, frame_bgr: np.ndarray, question: str = PROMPT, max_new_tokens: int = 48) -> str:
        from PIL import Image

        img = Image.fromarray(_prep(frame_bgr))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[img], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class HailoQwen2VL:
    """Mirrors ugen300-demos/V3/vlm_client.py (MIT) -- HailoRT GenAI VLM API."""

    name = "hailo:Qwen2-VL-2B-Instruct.hef"

    def __init__(self, hef_path: str = "models/Qwen2-VL-2B-Instruct.hef"):
        from hailo_platform.genai import VLM

        from .hailo_vendor import hailo_vdevice

        self.vlm = VLM(hailo_vdevice.get(), hef_path)
        hailo_vdevice.keep(self.vlm)

    def ask(self, frame_bgr: np.ndarray, question: str = PROMPT, max_new_tokens: int = 48) -> str:
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        resp = self.vlm.generate_all(prompt=prompt, frames=[_prep(frame_bgr)], temperature=0.1,
                                     seed=42, max_generated_tokens=max_new_tokens)
        self.vlm.clear_context()
        return resp if isinstance(resp, str) else str(resp)


def make_vlm(backend: str = "cpu", **kw):
    return HailoQwen2VL(**kw) if backend == "hailo" else HFQwen2VL(**kw)


def verify_fall(vlm, frame_bgr: np.ndarray) -> dict:
    raw = vlm.ask(frame_bgr)
    letter, text = parse_answer(raw)
    return {"choice": letter, "label": CHOICES.get(letter, "unknown"), "is_fall": letter in FALL_CHOICES or letter is None,  # unparseable -> fail safe
            "answer": text}
