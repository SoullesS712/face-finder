#!/usr/bin/env python3
"""รวม viewer + โปรเจกต์หนึ่งเป็นไฟล์ HTML เดียว (เปิด double-click ได้ ไม่ต้องมี server)
ใช้:  python make_standalone.py <slug> [output.html]
รูปถูกย่อเหลือกว้าง 900px เพื่อให้ไฟล์ไม่ใหญ่เกิน"""
import base64, json, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent / "viewer"
slug = sys.argv[1] if len(sys.argv) > 1 else "demo-ngan-buat"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"faceview_{slug}.html")

proj = ROOT / "projects" / slug
data = json.loads((proj / "data.json").read_text(encoding="utf-8"))
files = {}

def enc_img(path, max_w=None, q=72):
    img = cv2.imread(str(path))
    if img is None: return None
    if max_w and img.shape[1] > max_w:
        h = int(img.shape[0] * max_w / img.shape[1])
        img = cv2.resize(img, (max_w, h))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

for fp in sorted((proj / "faces").glob("*.jpg")):
    d = enc_img(fp, q=80)
    if d: files["faces/" + fp.name] = d
for ph in data["photos"]:
    if ph.get("src"):
        d = enc_img(proj / ph["src"], max_w=900)
        if d: files[ph["src"]] = d
        if ph.get("printSrc") == ph.get("src"):
            files[ph["printSrc"]] = files[ph["src"]]

inline = json.dumps({"data": data, "files": files}, ensure_ascii=False)
html = (ROOT / "index.html").read_text(encoding="utf-8")
html = html.replace("<script>", f"<script>window.__INLINE__={inline};</script>\n<script>", 1)
out.write_text(html, encoding="utf-8")
print(f"เขียน {out} ({out.stat().st_size/1e6:.1f} MB)")
