"""1280x720 dashboard renderer used for the demo video and the live viewer."""
from __future__ import annotations

import os
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (15, 20, 25)
PANEL = (27, 34, 42)
LINE = (52, 62, 74)
TEXT = (230, 236, 242)
MUTED = (140, 152, 166)
GREEN = (52, 199, 120)
AMBER = (255, 184, 48)
RED = (255, 82, 82)
BLUE = (88, 166, 255)
TEAL = (64, 210, 200)

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]

_F = os.environ.get("ANXIN_FONT", "/System/Library/Fonts/Avenir Next.ttc")


def font(size, w="medium"):
    idx = {"bold": 0, "demi": 2, "medium": 5, "regular": 7, "heavy": 8}[w]
    try:
        return ImageFont.truetype(_F, size, index=idx)
    except OSError:
        return ImageFont.load_default(size)


F = {k: font(*k) for k in [(30, "bold"), (22, "demi"), (18, "demi"), (17, "medium"), (15, "medium"), (14, "regular"),
                          (13, "medium"), (26, "demi"), (20, "medium"), (16, "regular")]}

STATE_COLOR = {"UPRIGHT": GREEN, "DESCENDING": AMBER, "DOWN": AMBER, "SUSPECTED_FALL": RED}


def draw_skeleton(img: np.ndarray, kps, scale=1.0, offset=(0, 0), color=(64, 210, 200), thick=3):
    if kps is None:
        return
    ox, oy = offset
    ok = kps[:, 2] >= 0.3
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            pa = (int(kps[a, 0] * scale + ox), int(kps[a, 1] * scale + oy))
            pb = (int(kps[b, 0] * scale + ox), int(kps[b, 1] * scale + oy))
            cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)
    for j in np.where(ok)[0]:
        cv2.circle(img, (int(kps[j, 0] * scale + ox), int(kps[j, 1] * scale + oy)), thick + 1, (255, 255, 255), -1, cv2.LINE_AA)


def _rr(d, box, fill, r=12, outline=None, width=1):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def _wrap(d, xy, text, f, fill, width_px, line_h=None):
    x, y = xy
    avg = f.getlength("abcdefghijklmnopqrstuvwxyz") / 26
    lines = textwrap.wrap(text, max(10, int(width_px / avg)))
    lh = line_h or int(f.size * 1.35)
    for ln in lines:
        d.text((x, y), ln, font=f, fill=fill)
        y += lh
    return y


def render(s: dict) -> np.ndarray:
    """s keys: frame, kps, feats, t, state, tiers(list of dicts), bubbles(list), notify(dict|None), caption, banner."""
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:] = BG[::-1]

    # camera panel (BGR work in OpenCV)
    cam_box = (24, 92, 664, 572)
    if s.get("frame") is not None:
        fr = s["frame"]
        sc = min(640 / fr.shape[1], 480 / fr.shape[0])
        view = cv2.resize(fr, (int(fr.shape[1] * sc), int(fr.shape[0] * sc)), interpolation=cv2.INTER_LINEAR)
        draw_skeleton(view, s.get("kps"), scale=sc, color=(200, 210, 64), thick=3)
        canvas[cam_box[1]:cam_box[1] + view.shape[0], cam_box[0]:cam_box[0] + view.shape[1]] = view
    # privacy panel: skeleton only
    pv = (688, 92, 968, 302)
    sk = np.zeros((pv[3] - pv[1], pv[2] - pv[0], 3), np.uint8)
    sk[:] = (20, 16, 12)
    if s.get("kps") is not None and s.get("frame") is not None:
        fr = s["frame"]
        sc2 = min(sk.shape[1] / fr.shape[1], sk.shape[0] / fr.shape[0])
        draw_skeleton(sk, s["kps"], scale=sc2, color=(200, 210, 64), thick=2)
    canvas[pv[1]:pv[3], pv[0]:pv[2]] = sk

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)

    # header
    d.text((24, 18), "AnXin Edge", font=F[(30, "bold")], fill=TEXT)
    d.text((212, 30), "privacy-first fall guardian  ·  on-device AI cascade", font=F[(17, "medium")], fill=MUTED)
    chip = s.get("backend", "Stage I: laptop CPU/MPS  ›  Stage II: ASUS UGen300 (Hailo-10H)")
    cw = F[(14, "regular")].getlength(chip) + 24
    _rr(d, (W - 24 - cw, 22, W - 24, 52), PANEL, r=15, outline=LINE)
    d.text((W - 12 - cw, 29), chip, font=F[(14, "regular")], fill=MUTED)

    d.text((24, 66), "Camera  ·  demo view only — frames are processed in memory, never stored or sent",
           font=F[(14, "regular")], fill=MUTED)
    d.rectangle(cam_box, outline=LINE, width=1)
    d.text((688, 66), "What the system keeps: skeleton only", font=F[(14, "regular")], fill=MUTED)
    d.rectangle(pv, outline=LINE, width=1)

    # feature readouts
    fx = 988
    st = s.get("state", "UPRIGHT")
    _rr(d, (fx, 92, 1256, 302), PANEL, outline=LINE)
    d.text((fx + 16, 102), "T1 posture state", font=F[(13, "medium")], fill=MUTED)
    d.text((fx + 16, 120), st.replace("_", " "), font=F[(22, "demi")], fill=STATE_COLOR.get(st, TEXT))
    f = s.get("feats") or {}
    rows = [("torso angle", f"{f['torso_angle']:.0f}°" if "torso_angle" in f else "—"),
            ("body aspect w/h", f"{f['aspect']:.2f}" if "aspect" in f else "—"),
            ("head height", f"{1 - f['top_y']:.2f}" if "top_y" in f else "—"),
            ("t", f"{s.get('t', 0):.2f} s")]
    y = 162
    for k, v in rows:
        d.text((fx + 16, y), k, font=F[(15, "medium")], fill=MUTED)
        d.text((fx + 250 - F[(18, "demi")].getlength(v), y - 2), v, font=F[(18, "demi")], fill=TEXT)
        y += 32

    # tier rows
    y = 318
    for tier in s.get("tiers", []):
        col = {"idle": LINE, "run": AMBER, "ok": GREEN, "alert": RED, "skip": LINE}[tier["status"]]
        _rr(d, (688, y, 1256, y + 58), PANEL, outline=col if tier["status"] != "idle" else LINE, width=2 if tier["status"] != "idle" else 1)
        d.ellipse((704, y + 22, 718, y + 36), fill=col)
        d.text((732, y + 7), tier["name"], font=F[(17, "medium")], fill=TEXT)
        d.text((732, y + 31), tier.get("detail", ""), font=F[(14, "regular")], fill=MUTED if tier["status"] == "idle" else TEXT)
        if tier.get("right"):
            rt = tier["right"]
            d.text((1240 - F[(14, "regular")].getlength(rt), y + 9), rt, font=F[(14, "regular")], fill=MUTED)
        y += 64

    # bottom strip: banner / bubbles / notification
    _rr(d, (24, 590, 1256, 704), PANEL, outline=LINE)
    if s.get("notify"):
        n = s["notify"]
        d.text((44, 600), n.get("title", "Message to family"), font=F[(13, "medium")], fill=n.get("color", RED))
        _wrap(d, (44, 622), n["text"], F[(17, "medium")], TEXT, 1180, 26)
    elif s.get("bubbles"):
        yy = 602
        for who, txt, col in s["bubbles"][-2:]:
            d.text((44, yy), who, font=F[(15, "medium")], fill=col)
            yy = _wrap(d, (200, yy), txt, F[(17, "medium")], TEXT, 1030, 24) + 10
    elif s.get("banner"):
        b = s["banner"]
        _wrap(d, (44, 604), b["text"], F[(20, "medium")], b.get("color", TEXT), 1180, 30)

    if s.get("disclaimer"):
        dtx = s["disclaimer"]
        dw = F[(13, "medium")].getlength(dtx) + 20
        _rr(d, (32, 100, 32 + dw, 124), (60, 44, 10), r=8, outline=AMBER)
        d.text((42, 104), dtx, font=F[(13, "medium")], fill=AMBER)

    if s.get("caption"):
        cap = s["caption"]
        tw = F[(20, "medium")].getlength(cap) + 32
        _rr(d, (344 - tw / 2, 520, 344 + tw / 2, 558), (0, 0, 0), r=10)
        d.text((360 - tw / 2, 526), cap, font=F[(20, "medium")], fill=(255, 255, 255))

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- guardian view
STEP_TEXT = {
    "vlm": "T1 suspected a fall — the on-device VLM is checking ONE keyframe (held in RAM) …",
    "ask": "Fall confirmed — AnXin is speaking to the person …",
    "listen": "Listening for a reply …",
    "understand": "Whisper is transcribing, the rules + LLM decide the intent …",
    "write": "LLM is writing the text message for the family …",
    "send": "Sending text-only alert to the family-phone receiver …",
}


def guardian_view(rec, st: dict, disclaimer: str | None = None, pose_ms: float | None = None, backend: str | None = None) -> dict:
    """Map a Guardian status (+ current FrameRecord) to dashboard render() input."""
    step, phase = st.get("step"), st.get("phase", "MONITORING")
    vlm = getattr(rec, "vlm", None) or st.get("last_vlm")
    t0 = {"name": "T0  Pose  ·  YOLOv8s-pose", "status": "ok", "detail": "17 keypoints per frame",
          "right": f"{pose_ms:.0f} ms/frame" if pose_ms else None}
    t1 = {"name": "T1  Skeleton fall state machine", "status": "idle", "detail": "watching for fast drop + stay down"}
    if rec.state in ("DESCENDING", "DOWN"):
        t1.update(status="run", detail="person is low / horizontal …")
    elif rec.state == "SUSPECTED_FALL":
        t1.update(status="alert", detail="suspected fall")
    t2 = {"name": "T2  VLM verify  ·  Qwen2-VL-2B", "status": "idle", "detail": "sleeps until T1 fires"}
    if step == "vlm":
        t2.update(status="run", detail="checking one keyframe …")
    elif vlm is not None:
        t2.update(status="alert" if vlm["is_fall"] else "ok",
                  detail=f"“{vlm['choice']}) {vlm['label']}”  ›  " + ("confirmed" if vlm["is_fall"] else "dismissed"),
                  right=f"{vlm['ms']:.0f} ms" if vlm.get("ms") else None)
    t3 = {"name": "T3  Voice check-in  ·  Whisper-Base + Qwen2.5-1.5B", "status": "idle", "detail": "talks to the person, decides escalation"}
    if step in ("ask", "listen", "understand", "write", "send"):
        t3.update(status="run", detail=STEP_TEXT[step].split("—")[-1].strip())
    if st.get("intent"):
        t3.update(status="ok" if st["intent"] == "OK" else "alert",
                  detail=f"heard “{st.get('transcript', '')[:48]}”  ›  {st['intent']} (by {st.get('decided_by')})")
    if phase == "WATCHING":
        t3["right"] = f"long-lie timer {st.get('watch_left_s', 0):.1f} s"
    out = {"frame": rec.frame, "kps": rec.kps, "feats": rec.feats, "t": rec.t, "state": rec.state,
           "tiers": [t0, t1, t2, t3], "disclaimer": disclaimer}
    if backend:
        out["backend"] = backend
    alerts = st.get("alerts") or []
    if step in STEP_TEXT:
        out["banner"] = {"text": STEP_TEXT[step], "color": AMBER}
    elif alerts:
        main = [x for x in alerts if x["level"] in ("URGENT", "FYI")][-1]
        detail = alerts[-1]["message"] if alerts[-1]["kind"] == "details" else None
        r, tm = main.get("receipt", {}), main.get("timings", {})
        when = f"  ·  {tm['t1_to_delivered_s']:.1f} s after T1 suspicion" if "t1_to_delivered_s" in tm else ""
        rcpt = (f"DELIVERED to family-phone receiver (ack #{r.get('ack', {}).get('id')}, {r.get('rtt_ms')} ms){when}"
                if r.get("delivered") else f"NOT delivered ({r.get('error')}) — local siren sounded")
        if detail:
            d = detail.removeprefix("Details: ")
            d = d if len(d) <= 150 else d[:147].rsplit(" ", 1)[0] + " …"
        text = main["message"] + (f"   ›  LLM follow-up: {d}" if detail else "")
        out["notify"] = {"title": f"{main['level']} · {main['kind']}  ·  {rcpt}", "text": text,
                         "color": RED if main["level"] == "URGENT" else AMBER}
    elif step == "dismissed" and vlm is not None:
        out["banner"] = {"text": f"VLM: “{vlm['label']}” › false alarm DISMISSED. No alert, nobody disturbed.", "color": GREEN}
    else:
        out["banner"] = {"text": "Monitoring…  only the skeleton on the right is kept."}
    if phase == "WATCHING" and "notify" in out:
        out["notify"]["title"] += f"   ·   long-lie timer: {st.get('watch_left_s', 0):.1f} s left"
    return out
