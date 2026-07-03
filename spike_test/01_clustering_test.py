#!/usr/bin/env python3
"""
Spike 1: ทดสอบความแม่นของ clustering กับรูปจริง
วิธีใช้:  python 01_clustering_test.py ./โฟลเดอร์รูปตัวอย่าง [--eps 0.55]
ผลลัพธ์: spike1_report.html  เปิดดูใน browser -> เห็นแต่ละกลุ่มว่ารวมถูกคนไหม

เกณฑ์ตัดสิน go/no-go:
  - คนหลัก (นาค/พ่อ/แม่) แตกเป็น <= 3 กลุ่ม  -> ผ่าน (merge มือได้)
  - หน้าในภาพหมู่จับได้ > 80%                 -> ผ่าน
  - คนละคนถูกรวมกลุ่มเดียวกัน (false merge)   -> ถ้าเกิน 5% ให้ลด --eps
"""
import argparse, base64, sys
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--eps", type=float, default=0.55)
ap.add_argument("--min-face", type=int, default=36)
args = ap.parse_args()

try:
    from insightface.app import FaceAnalysis
    from sklearn.cluster import DBSCAN
except ImportError:
    sys.exit("pip install insightface onnxruntime scikit-learn opencv-python")

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(1280, 1280))

imgs = sorted(p for p in Path(args.folder).iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
print(f"พบ {len(imgs)} รูป")

embs, crops, srcs = [], [], []
for p in imgs:
    im = cv2.imread(str(p))
    if im is None: continue
    faces = app.get(im)
    kept = 0
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        if min(x2-x1, y2-y1) < args.min_face or f.det_score < 0.55: continue
        pad = int(.25*max(x2-x1, y2-y1)); H, W = im.shape[:2]
        c = im[max(0,y1-pad):min(H,y2+pad), max(0,x1-pad):min(W,x2+pad)]
        c = cv2.resize(c, (120, 120))
        embs.append(f.normed_embedding); crops.append(c); srcs.append(p.name); kept += 1
    print(f"  {p.name}: หน้า {kept}")

labels = DBSCAN(eps=args.eps, min_samples=2, metric="cosine").fit_predict(np.vstack(embs))
groups = {}
for i, l in enumerate(labels): groups.setdefault(int(l), []).append(i)
print(f"\nรวม {len(embs)} หน้า -> {len([k for k in groups if k>=0])} กลุ่ม + noise {len(groups.get(-1,[]))} หน้า")

def b64(img):
    return base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()

rows = []
for l in sorted(groups, key=lambda k: (-len(groups[k]) if k >= 0 else 1e9)):
    title = f"กลุ่ม {l} ({len(groups[l])} หน้า)" if l >= 0 else f"Noise/เดี่ยว ({len(groups[l])} หน้า)"
    cells = "".join(
        f'<figure><img src="data:image/jpeg;base64,{b64(crops[i])}"><figcaption>{srcs[i]}</figcaption></figure>'
        for i in groups[l])
    rows.append(f"<h2>{title}</h2><div class=g>{cells}</div>")

Path("spike1_report.html").write_text(f"""<!DOCTYPE html><meta charset=utf8>
<title>Spike1 eps={args.eps}</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee;padding:20px}}
.g{{display:flex;flex-wrap:wrap;gap:6px}}figure{{margin:0;text-align:center}}
img{{border-radius:6px}}figcaption{{font-size:9px;color:#999;max-width:120px;overflow:hidden}}
h2{{margin:24px 0 8px;font-size:15px;color:#d9a441}}</style>
<h1>ผล clustering (eps={args.eps}, {len(imgs)} รูป, {len(embs)} หน้า)</h1>
<p>ตรวจตา: กลุ่มเดียวกันต้องเป็นคนเดียวกัน / คนหลักไม่ควรแตกเกิน 3 กลุ่ม</p>
{''.join(rows)}""", encoding="utf-8")
print("เขียน spike1_report.html แล้ว เปิดดูใน browser ได้เลย")
