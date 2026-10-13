import threading
from http.server import ThreadingHTTPServer

import scripts.family_phone as fp
from anxin.notify import Notifier


def test_delivery_to_receiver_and_ack(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "LOG", tmp_path / "log.jsonl")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), fp.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    r = Notifier(f"http://127.0.0.1:{srv.server_port}/alert").send({"level": "URGENT", "kind": "t", "message": "hi"})
    srv.shutdown()
    assert r["delivered"] and r["ack"]["ok"] and r["ack"]["id"] >= 1
    assert "hi" in (tmp_path / "log.jsonl").read_text()


def test_failed_delivery_sounds_local_siren():
    said = []
    r = Notifier("http://127.0.0.1:9/alert", retries=2, timeout=0.2, local_siren=said.append).send(
        {"level": "URGENT", "kind": "t", "message": "x"})
    assert not r["delivered"] and said
