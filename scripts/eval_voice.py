"""Voice check-in evaluation: scripted replies x 3 TTS voices -> Whisper-Base -> rules + LLM.

Measures the two error types that matter:
  missed_help  : a HELP reply classified OK      (dangerous: would only send an FYI)
  false_urgent : an OK reply escalated to URGENT (nuisance alert)
UNCLEAR / NO_RESPONSE are escalated by design. Synthetic speech is a limitation: real,
elderly, accented and Mandarin/Taiwanese speech must be tested in Stage II.
"""
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anxin.backends.genai import make_asr, make_llm  # noqa: E402
from anxin.checkin import classify, decide, load_wav, speak  # noqa: E402

UTTERANCES = {
    "HELP": ["I can't get up.", "My hip really hurts.", "Help me, please.", "I think I broke my wrist.",
             "Call my daughter, I'm hurt.", "I'm bleeding.", "I feel dizzy, I can't stand.", "Ow, my back, it hurts so much.",
             "Please call an ambulance.", "I'm not okay.", "I can't move my leg.", "I'm not hurt, but I can't get up."],
    "OK": ["I'm fine, I just slipped.", "I'm okay, give me a minute.", "Yes, yes, I'm alright.", "I don't need help, I'm okay.",
           "It doesn't hurt, I'm fine.", "No, no, I'm good, just picking up my keys.", "I'm okay dear, don't worry.",
           "I'm fine, I was just doing my exercises.", "All good, I'm getting up now.", "No pain, I'm alright."],
    "UNCLEAR": ["Where are my glasses?", "What time is it?", "Who is that?", "Hmm?", "Is it dinner yet?",
                "What happened?"],
}
VOICES = ["Grandma (English (US))", "Grandpa (English (US))", "Daniel"]


def main():
    asr, llm = make_asr("cpu"), make_llm("cpu")
    tmp = Path(tempfile.mkdtemp())
    rows = []
    for label, texts in UTTERANCES.items():
        for t in texts:
            for v in VOICES:
                wav = speak(t, tmp / f"{len(rows)}.wav", v, rate=150)
                heard = asr.transcribe(load_wav(wav))
                intent, by = classify(heard, llm)
                rows.append({"label": label, "text": t, "voice": v, "heard": heard, "intent": intent, "by": by,
                             "action": decide(intent)})
                print(label, "|", t, "|", v.split()[0], "->", intent, by, "| heard:", heard, flush=True)
    n = len(rows)
    missed_help = [r for r in rows if r["label"] == "HELP" and r["action"] != "ALERT_URGENT"]
    false_urgent = [r for r in rows if r["label"] == "OK" and r["action"] == "ALERT_URGENT"]
    unclear_ok = [r for r in rows if r["label"] == "UNCLEAR" and r["action"] != "ALERT_URGENT"]
    summ = {
        "samples": n, "voices": VOICES,
        "help_escalated": f"{sum(r['label'] == 'HELP' for r in rows) - len(missed_help)}/{sum(r['label'] == 'HELP' for r in rows)}",
        "ok_kept_as_fyi": f"{sum(r['label'] == 'OK' for r in rows) - len(false_urgent)}/{sum(r['label'] == 'OK' for r in rows)}",
        "unclear_escalated": f"{sum(r['label'] == 'UNCLEAR' for r in rows) - len(unclear_ok)}/{sum(r['label'] == 'UNCLEAR' for r in rows)}",
        "decided_by": dict(Counter(r["by"] for r in rows)),
        "missed_help": missed_help, "false_urgent": false_urgent, "unclear_not_escalated": unclear_ok,
    }
    (ROOT / "outputs/eval_voice.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, list)}, indent=2))
    for k in ("missed_help", "false_urgent", "unclear_not_escalated"):
        for r in summ[k]:
            print(k, "|", r["text"], "| heard:", r["heard"], "|", r["intent"], r["by"])


if __name__ == "__main__":
    main()
