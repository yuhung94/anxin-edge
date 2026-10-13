"""Tier-3 speech + language backends.

    ASR : Whisper-Base           cpu -> openai-whisper   | hailo -> Whisper-Base.hef (Speech2Text)
    LLM : Qwen2.5-1.5B-Instruct  cpu -> HF transformers  | hailo -> Qwen2.5-1.5B-Instruct.hef (genai.LLM)

Same model sizes on both paths, so Stage I behaviour carries over to the UGen300.
"""
from __future__ import annotations

import numpy as np

SR = 16000


# ---------------------------------------------------------------- ASR
class WhisperCPU:
    name = "cpu:whisper-base"

    def __init__(self, size: str = "base"):
        import whisper

        self.model = whisper.load_model(size, device="cpu")

    def transcribe(self, audio_f32: np.ndarray, language: str | None = None) -> str:
        return self.model.transcribe(audio_f32, language=language, fp16=False)["text"].strip()


class WhisperHailo:
    name = "hailo:Whisper-Base.hef"

    def __init__(self, hef_path: str = "models/Whisper-Base.hef"):
        from hailo_platform.genai import Speech2Text

        from .hailo_vendor import hailo_vdevice

        self.stt = Speech2Text(hailo_vdevice.get(), hef_path)
        hailo_vdevice.keep(self.stt)

    def transcribe(self, audio_f32: np.ndarray, language: str | None = None) -> str:
        from hailo_platform.genai import Speech2TextTask

        kw = {"audio_data": audio_f32, "task": Speech2TextTask.TRANSCRIBE, "timeout_ms": 15000}
        if language:
            kw["language"] = language
        segs = self.stt.generate_all_segments(**kw)
        return " ".join(getattr(s, "text", str(s)) for s in segs).strip()


# ---------------------------------------------------------------- LLM
class QwenCPU:
    name = "cpu:Qwen2.5-1.5B-Instruct"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()

    def chat(self, system: str, user: str, max_new_tokens: int = 160) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class QwenHailo:
    """Mirrors ugen300-demos/WQ1/summarize.py (MIT) -- HailoRT GenAI LLM API."""

    name = "hailo:Qwen2.5-1.5B-Instruct.hef"

    def __init__(self, hef_path: str = "models/Qwen2.5-1.5B-Instruct.hef"):
        from hailo_platform.genai import LLM

        from .hailo_vendor import hailo_vdevice

        self.llm = LLM(hailo_vdevice.get(), hef_path)
        hailo_vdevice.keep(self.llm)

    def chat(self, system: str, user: str, max_new_tokens: int = 160) -> str:
        prompt = [{"role": "system", "content": [{"type": "text", "text": system}]},
                  {"role": "user", "content": [{"type": "text", "text": user}]}]
        resp = self.llm.generate_all(prompt=prompt, temperature=0.1, seed=42, max_generated_tokens=max_new_tokens)
        self.llm.clear_context()
        return (resp if isinstance(resp, str) else str(resp)).strip()


def make_asr(backend: str = "cpu", **kw):
    return WhisperHailo(**kw) if backend == "hailo" else WhisperCPU(**kw)


def make_llm(backend: str = "cpu", **kw):
    return QwenHailo(**kw) if backend == "hailo" else QwenCPU(**kw)
