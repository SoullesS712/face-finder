#!/usr/bin/env python3
"""
Spike 3: เช็ค mapping ชื่อไฟล์ระหว่างโฟลเดอร์ facebook กับโฟลเดอร์ปริ้น
วิธีใช้:
  python 03_filename_mapping.py --api-key AIza... \
    --fb "https://drive.google.com/drive/folders/1jP8jTxKH2-pdKMxs7qngFSFGsKkjEBgh" \
    --pr "https://drive.google.com/drive/folders/1PtGX6MRmpUN5E6lx9-MXDPgVwedM7zUL"

พร้อมกันนี้จะ generate spike2_embed_test.html ที่ฝังรูปจริง 6 รูปแรกจาก Drive
เปิดใน browser -> ถ้ารูปขึ้นครบ = spike 2 ผ่านด้วย
"""
import argparse, json, re, sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

ap = argparse.ArgumentParser()
ap.add_argument("--api-key", required=True)
ap.add_argument("--fb", required=True)
ap.add_argument("--pr", required=True)
args = ap.parse_args()

fid = lambda s: re.search(r"folders/([\w-]+)", s).group(1) if "folders/" in s else s

def list_folder(folder_id):
    files, tok = [], None
    while True:
        params = {"q": f"'{folder_id}' in parents and trashed=false",
                  "fields": "nextPageToken, files(id,name)", "pageSize": 1000,
                  "key": args.api_key}
        if tok: params["pageToken"] = tok
        r = requests.get("https://www.googleapis.com/drive/v3/files", params=params, timeout=30)
        if r.status_code != 200:
            sys.exit(f"Drive API {r.status_code}: {r.text[:300]}")
        d = r.json(); files += d.get("files", []); tok = d.get("nextPageToken")
        if not tok: return files

fb = list_folder(fid(args.fb))
pr = list_folder(fid(args.pr))
fb_stems = {Path(f["name"]).stem: f for f in fb}
pr_stems = {Path(f["name"]).stem: f for f in pr}

matched = fb_stems.keys() & pr_stems.keys()
only_fb = fb_stems.keys() - pr_stems.keys()
only_pr = pr_stems.keys() - fb_stems.keys()

print(f"facebook: {len(fb)} ไฟล์ | ปริ้น: {len(pr)} ไฟล์")
print(f"map ตรงกัน: {len(matched)} ({len(matched)/max(1,len(fb))*100:.1f}% ของ fb)")
if only_fb: print(f"มีเฉพาะ fb {len(only_fb)} เช่น {sorted(only_fb)[:5]}")
if only_pr: print(f"มีเฉพาะปริ้น {len(only_pr)} เช่น {sorted(only_pr)[:5]}")
print("=> " + ("ผ่าน: ใช้ชื่อไฟล์ map ได้" if len(matched)/max(1,len(fb)) > 0.95
      else "เตือน: ชื่อไม่ตรงเกิน 5% ต้องดู pattern เพิ่ม"))

# ---- spike 2: สร้างหน้า embed test จากรูปจริง 6 รูปแรก ----
cells = "".join(
    f'<div><img src="https://drive.google.com/thumbnail?id={f["id"]}&sz=w400">'
    f'<p>{f["name"]}</p></div>' for f in fb[:6])
Path("spike2_embed_test.html").write_text(f"""<!DOCTYPE html><meta charset=utf8>
<title>Spike2 Drive embed</title>
<style>body{{background:#111;color:#eee;font-family:sans-serif;display:flex;flex-wrap:wrap;gap:10px;padding:20px}}
img{{width:300px;border-radius:8px}}p{{font-size:11px;color:#999}}</style>
<h1 style="width:100%">ถ้ารูปขึ้นครบ 6 รูป = Drive embed ใช้ได้ ✔</h1>{cells}""",
    encoding="utf-8")
print("เขียน spike2_embed_test.html แล้ว เปิดใน browser เพื่อเช็ค embed")
