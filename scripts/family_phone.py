"""Simulated family phone: a separate process that RECEIVES AnXin alerts over HTTP.

    python scripts/family_phone.py            # then open http://127.0.0.1:8799

POST /alert  -> stores the alert, returns an ack {ok, id, received_at}
GET  /alerts -> JSON list;  GET / -> phone-style page (auto-refresh)
Every receipt is appended to outputs/family_phone_log.jsonl (delivery evidence).
In a deployment this endpoint is replaced by Telegram/LINE or a care-provider server.
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "outputs/family_phone_log.jsonl"
ALERTS, LOCK = [], threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Family phone</title><style>
body{margin:0;background:#1b2b2e;font-family:-apple-system,Helvetica,Arial;display:flex;justify-content:center}
.phone{width:360px;min-height:640px;margin:24px;background:#f4f7f6;border-radius:36px;padding:28px 18px;box-shadow:0 10px 40px #0008}
h1{font-size:15px;color:#6c7f82;text-align:center;margin:0 0 16px}
.card{background:#fff;border-radius:16px;padding:14px 16px;margin-bottom:12px;box-shadow:0 1px 4px #0002}
.lvl{font-weight:700;font-size:13px}.URGENT{color:#e8684a}.FYI{color:#c48a12}.INFO{color:#1f8a84}
.msg{font-size:15px;color:#1b2b2e;margin:6px 0}.meta{font-size:11px;color:#6c7f82}
.empty{text-align:center;color:#6c7f82;margin-top:120px}</style></head><body><div class="phone">
<h1>AnXin · Family phone (receiver)</h1><div id="list"><div class="empty">No alerts</div></div></div>
<script>
async function poll(){try{const a=await (await fetch('/alerts')).json();
if(a.length){document.getElementById('list').innerHTML=a.slice().reverse().map(x=>
`<div class="card"><div class="lvl ${x.level}">${x.level} · ${x.kind}</div><div class="msg">${x.message}</div>
<div class="meta">received ${new Date(x.received_at*1000).toLocaleTimeString()} · id ${x.id}</div></div>`).join('')}}catch(e){}}
setInterval(poll,1000);poll();</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/alerts":
            with LOCK:
                self._send(200, json.dumps(ALERTS))
        else:
            self._send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/alert":
            return self._send(404, "{}")
        n = int(self.headers.get("Content-Length", 0))
        try:
            alert = json.loads(self.rfile.read(n))
        except json.JSONDecodeError:
            return self._send(400, '{"ok": false}')
        with LOCK:
            alert["id"], alert["received_at"] = len(ALERTS) + 1, time.time()
            ALERTS.append(alert)
            LOG.parent.mkdir(exist_ok=True)
            with LOG.open("a") as f:
                f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        print(f"[family phone] {alert.get('kind')}: {alert.get('message')}", flush=True)
        self._send(200, json.dumps({"ok": True, "id": alert["id"], "received_at": alert["received_at"]}))


def serve(port=8799):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"family phone listening on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8799)
