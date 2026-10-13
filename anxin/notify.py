"""Alert delivery. Only JSON text is sent -- never images or audio.

Channels
  webhook  : POST JSON to ANXIN_WEBHOOK (default: the local family-phone receiver,
             scripts/family_phone.py). Delivery = HTTP 200 + JSON ack from the receiver.
  telegram : set ANXIN_TELEGRAM_TOKEN + ANXIN_TELEGRAM_CHAT (implemented, not yet field-tested).
Every send is retried 3x with back-off; if all attempts fail AnXin sounds a local spoken
siren so the person nearby still gets help, and the failure is reported in the receipt.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_WEBHOOK = "http://127.0.0.1:8799/alert"


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode() or "{}"
        if r.status != 200:
            raise urllib.error.HTTPError(url, r.status, body, r.headers, None)
        return json.loads(body)


class Notifier:
    def __init__(self, webhook: str | None = None, retries: int = 3, timeout: float = 3.0, local_siren=None):
        self.webhook = webhook or os.environ.get("ANXIN_WEBHOOK", DEFAULT_WEBHOOK)
        self.tg_token = os.environ.get("ANXIN_TELEGRAM_TOKEN")
        self.tg_chat = os.environ.get("ANXIN_TELEGRAM_CHAT")
        self.retries, self.timeout = retries, timeout
        self.local_siren = local_siren  # callable(text) used when delivery fails

    def _send_once(self, alert: dict) -> dict:
        if self.tg_token and self.tg_chat:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            ack = _post(url, {"chat_id": self.tg_chat, "text": f"{alert['level']}: {alert['message']}"}, self.timeout)
            return {"channel": "telegram", "ack": {"ok": ack.get("ok")}}
        return {"channel": "webhook", "ack": _post(self.webhook, alert, self.timeout)}

    def send(self, alert: dict) -> dict:
        err = None
        for attempt in range(1, self.retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._send_once(alert)
                return {"delivered": True, "attempts": attempt, "rtt_ms": round(1000 * (time.perf_counter() - t0), 1), **r}
            except Exception as e:  # noqa: BLE001 - any transport error triggers retry
                err = repr(e)
                time.sleep(0.5 * attempt)
        if self.local_siren:
            self.local_siren("Alert could not be sent. If you need help, please call for help.")
        return {"delivered": False, "attempts": self.retries, "error": err, "channel": "none (local siren)"}
