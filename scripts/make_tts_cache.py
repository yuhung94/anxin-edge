"""Pre-synthesise the fixed check-in prompt (macOS `say`) so Linux/Pi hosts reuse the same voice."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anxin.audio import TTS_DIR  # noqa: E402
from anxin.checkin import CHECKIN_PROMPT, speak  # noqa: E402

VOICE = "Karen"
TTS_DIR.mkdir(parents=True, exist_ok=True)
for text in [CHECKIN_PROMPT, "Alert could not be sent. If you need help, please call for help."]:
    out = TTS_DIR / f"{hashlib.sha1((VOICE + text).encode()).hexdigest()[:12]}.wav"
    speak(text, out, VOICE)
    print(out.name, text)
