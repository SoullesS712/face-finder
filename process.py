#!/usr/bin/env python3
"""
Face Finder - Processing Pipeline
รันบนเครื่องช่างภาพ/แอดมิน ครั้งเดียวต่องาน แล้วได้ static data ไปวางบนเว็บ

โหมดการใช้งาน:

A) จาก Google Drive (แนะนำ - ได้ลิงก์ดาวน์โหลดไฟล์ปริ้นอัตโนมัติ):
   python process.py --project "งานบวชนัท" \
       --fb-folder "https://drive.google.com/drive/folders/XXXX" \
       --print-folder "https://drive.google.com/drive/folders/YYYY" \
       --api-key "AIza..."

   วิธีสร้าง API key (ฟรี, 2 นาที):
   1. https://console.cloud.google.com/ -> สร้าง project
   2. APIs & Services -> Enable "Google Drive API"
   3. Credentials -> Create credentials -> API key
   หมายเหตุ: โฟลเดอร์ต้อง share เป็น "Anyone with the link"

B) จากโฟลเดอร์ในเครื่อง (ไม่ต้องมี API key แต่ viewer จะใช้รูป local):
   python process.py --project "งานบวชนัท" \
       --fb-dir ./facebook --print-dir ./large

ติดตั้ง dependencies:
   pip install insightface onnxruntime opencv-python scikit-learn requests tqdm
"""

import argparse
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------- utils

def slugify(name: str) -> str:
    """ทำชื่อโปรเจกต์เป็น slug ปลอดภัยสำหรับ URL/โฟลเดอร์ (รองรับไทย -> translit เป็น hex ถ้าจำเป็น)"""
    s = unicodedata.normalize("NFKC", name.strip())
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w\-ก-๙]", "", s)
    if not s:
        s = "project"
    return s


def log(msg: str):
    print(f"[face-finder] {msg}", flush=True)


# ---------------------------------------------------------------- drive api

DRIVE_LIST_URL = "https://www.googleapis.com/drive/v3/files"
FOLDER_ID_RE = re.compile(r"folders/([A-Za-z0-9_-]+)")


def extract_folder_id(link_or_id: str) -> str:
    m = FOLDER_ID_RE.search(link_or_id)
    return m.group(1) if m else link_or_id.strip()


def drive_list_folder(folder_id: str, api_key: str) -> list[dict]:
    """คืน [{id, name, mimeType}] ของไฟล์ทั้งหมดในโฟลเดอร์ (public + API key)"""
    import requests

    files, page_token = [], None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(id,name,mimeType,size)",
            "pageSize": 1000,
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(DRIVE_LIST_URL, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(
                f"Drive API error {r.status_code}: {r.text[:300]}\n"
                "เช็คว่า 1) เปิดใช้ Drive API แล้ว 2) โฟลเดอร์ share เป็น anyone with link"
            )
        data = r.json()
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return [f for f in files if Path(f["name"]).suffix.lower() in IMAGE_EXTS]


def drive_download(file_id: str, api_key: str, dest: Path):
    import time

    import requests

    url = f"{DRIVE_LIST_URL}/{file_id}"
    for attempt in range(4):
        r = requests.get(url, params={"alt": "media", "key": api_key}, timeout=120)
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        dest.write_bytes(r.content)
        return
    # alt=media โดน rate limit ถาวรกับบางไฟล์ -> ใช้ thumbnail endpoint (w1600 พอสำหรับ detect)
    thumb = f"https://drive.google.com/thumbnail?id={file_id}&sz=w1600"
    for attempt in range(4):
        r = requests.get(thumb, timeout=120, allow_redirects=True)
        if r.status_code == 200 and r.content[:3] == b"\xff\xd8\xff":
            dest.write_bytes(r.content)
            return
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"โหลดไม่สำเร็จ (ทั้ง alt=media และ thumbnail): {file_id}")


# ---------------------------------------------------------------- face engine

def load_face_app(det_size: int = 1280):
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        sys.exit("ยังไม่ได้ติดตั้ง insightface: pip install insightface onnxruntime")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def detect_faces(app, img_bgr: np.ndarray, min_face_px: int, min_det_score: float):
    """คืน list ของ (bbox, embedding, det_score) ที่ผ่านเกณฑ์คุณภาพ"""
    faces = app.get(img_bgr)
    results = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        if min(w, h) < min_face_px:
            continue
        if f.det_score < min_det_score:
            continue
        emb = f.normed_embedding.astype(np.float32)
        results.append(((x1, y1, x2, y2), emb, float(f.det_score)))
    return results


def cluster_embeddings(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN บน cosine distance ของ normalized embeddings คืน labels (-1 = noise)"""
    from sklearn.cluster import DBSCAN

    if len(embeddings) == 0:
        return np.array([], dtype=int)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return db.fit_predict(embeddings)


# ---------------------------------------------------------------- pipeline

def main():
    ap = argparse.ArgumentParser(description="Face Finder processing pipeline")
    ap.add_argument("--project", required=True, help="ชื่องาน เช่น 'งานบวชนัท'")
    ap.add_argument("--fb-folder", help="ลิงก์ Drive โฟลเดอร์รูป facebook (ไฟล์เล็ก)")
    ap.add_argument("--print-folder", help="ลิงก์ Drive โฟลเดอร์รูปปริ้น (ไฟล์ใหญ่)")
    ap.add_argument("--api-key", help="Google Drive API key")
    ap.add_argument("--fb-dir", help="โฟลเดอร์ local รูป facebook (โหมด B)")
    ap.add_argument("--print-dir", help="โฟลเดอร์ local รูปปริ้น (โหมด B)")
    ap.add_argument("--out", default="viewer", help="โฟลเดอร์เว็บ viewer (default: viewer)")
    ap.add_argument("--min-face", type=int, default=36, help="ขนาดหน้าขั้นต่ำ px (default 36)")
    ap.add_argument("--min-score", type=float, default=0.55, help="det score ขั้นต่ำ (default 0.55)")
    ap.add_argument("--eps", type=float, default=0.55, help="DBSCAN eps, มากขึ้น=รวมกลุ่มง่ายขึ้น (default 0.55)")
    ap.add_argument("--min-samples", type=int, default=2, help="จำนวนหน้าขั้นต่ำต่อกลุ่ม (default 2)")
    ap.add_argument("--det-size", type=int, default=1280, help="detection resolution (default 1280)")
    ap.add_argument("--max-photos", type=int, default=1000)
    args = ap.parse_args()

    drive_mode = bool(args.fb_folder and args.api_key)
    local_mode = bool(args.fb_dir)
    if not drive_mode and not local_mode:
        ap.error("ต้องระบุ (--fb-folder + --api-key) หรือ --fb-dir อย่างใดอย่างหนึ่ง")

    slug = slugify(args.project)
    out_root = Path(args.out)
    proj_dir = out_root / "projects" / slug
    faces_dir = proj_dir / "faces"
    cache_dir = Path(".cache") / slug
    for d in (faces_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------- 1) รวบรวมรายการรูป + mapping fb <-> print ----------
    photos = []  # {name, fb_local, fb_drive_id?, print_drive_id?, print_local?}

    if drive_mode:
        log("ดึงรายชื่อไฟล์จาก Google Drive...")
        fb_files = drive_list_folder(extract_folder_id(args.fb_folder), args.api_key)
        log(f"  โฟลเดอร์ facebook: {len(fb_files)} รูป")
        print_map = {}
        if args.print_folder:
            pr_files = drive_list_folder(extract_folder_id(args.print_folder), args.api_key)
            log(f"  โฟลเดอร์ปริ้น: {len(pr_files)} รูป")
            print_map = {Path(f["name"]).stem: f["id"] for f in pr_files}

        fb_files = fb_files[: args.max_photos]
        unmatched = 0
        for f in fb_files:
            stem = Path(f["name"]).stem
            pid = print_map.get(stem)
            if args.print_folder and pid is None:
                unmatched += 1
            photos.append({
                "name": f["name"], "fb_drive_id": f["id"],
                "print_drive_id": pid, "fb_local": cache_dir / f["name"],
            })
        if args.print_folder:
            log(f"  mapping ด้วยชื่อไฟล์: ตรงกัน {len(photos)-unmatched}/{len(photos)} รูป"
                + (f" (ไม่พบคู่ปริ้น {unmatched} รูป)" if unmatched else " ✔"))

        log("ดาวน์โหลดรูป facebook (เก็บ cache ไว้ รันซ้ำไม่โหลดใหม่)...")
        try:
            from tqdm import tqdm
            iterator = tqdm(photos, unit="img")
        except ImportError:
            iterator = photos
        for p in iterator:
            if not p["fb_local"].exists():
                drive_download(p["fb_drive_id"], args.api_key, p["fb_local"])
    else:
        fb_dir = Path(args.fb_dir)
        names = sorted([p for p in fb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        names = names[: args.max_photos]
        print_map = {}
        if args.print_dir:
            pdir = Path(args.print_dir)
            print_map = {p.stem: p for p in pdir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        media_dir = proj_dir / "media"
        media_dir.mkdir(exist_ok=True)
        for src in names:
            dst = media_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            photos.append({
                "name": src.name, "fb_local": dst,
                "fb_drive_id": None,
                "print_drive_id": None,
                "print_local": print_map.get(src.stem),
            })
        log(f"โหมด local: {len(photos)} รูป, mapping ปริ้นตรงกัน "
            f"{sum(1 for p in photos if p.get('print_local'))}/{len(photos)}")

    if not photos:
        sys.exit("ไม่พบรูปในโฟลเดอร์")

    # ---------- 2) detect + embed ----------
    log(f"โหลดโมเดล InsightFace (buffalo_l, det_size={args.det_size})...")
    app = load_face_app(args.det_size)

    all_embs, face_meta = [], []  # face_meta: (photo_idx, bbox, score)
    log(f"ตรวจจับใบหน้า {len(photos)} รูป (min_face={args.min_face}px, min_score={args.min_score})...")
    try:
        from tqdm import tqdm
        iterator = enumerate(tqdm(photos, unit="img"))
    except ImportError:
        iterator = enumerate(photos)

    def make_crop(img, bbox):
        """crop หน้า + ขยายกรอบ 25% แล้วย่อเหลือกว้าง 160px"""
        x1, y1, x2, y2 = bbox
        pad = int(0.25 * max(x2 - x1, y2 - y1))
        H, W = img.shape[:2]
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
        crop = img[cy1:cy2, cx1:cx2]
        h = int(160 * crop.shape[0] / max(1, crop.shape[1]))
        return cv2.resize(crop, (160, max(1, h)))

    crops = []
    for i, p in iterator:
        img = cv2.imread(str(p["fb_local"]))
        if img is None:
            log(f"  ! อ่านไฟล์ไม่ได้ ข้าม: {p['name']}")
            continue
        p["w"], p["h"] = img.shape[1], img.shape[0]
        for bbox, emb, score in detect_faces(app, img, args.min_face, args.min_score):
            all_embs.append(emb)
            face_meta.append((i, bbox, score))
            crops.append(make_crop(img, bbox))  # crop ทันที ไม่ถือรูปทั้งชุดในแรม

    log(f"พบใบหน้าทั้งหมด {len(all_embs)} หน้า")

    # ---------- 3) clustering ----------
    log(f"จัดกลุ่มใบหน้า (DBSCAN eps={args.eps}, min_samples={args.min_samples})...")
    embs = np.vstack(all_embs) if all_embs else np.zeros((0, 512))
    labels = cluster_embeddings(embs, args.eps, args.min_samples)

    # noise (-1) -> แยกเป็น singleton คนละกลุ่ม เพื่อไม่ให้หน้าหาย
    next_label = labels.max() + 1 if len(labels) and labels.max() >= 0 else 0
    for j in range(len(labels)):
        if labels[j] == -1:
            labels[j] = next_label
            next_label += 1

    n_people = len(set(labels.tolist()))
    log(f"ได้ {n_people} กลุ่ม (คน) จาก {len(labels)} หน้า")

    # ---------- 4) export ----------
    log("บันทึกผลลัพธ์...")
    people: dict[int, dict] = {}
    photo_faces: dict[int, list] = {}

    for j, (pi, bbox, score) in enumerate(face_meta):
        lab = int(labels[j])
        person = people.setdefault(lab, {"faces": [], "photoIdx": set(), "best": (-1, None)})
        face_id = f"f{j:05d}"
        person["faces"].append(face_id)
        person["photoIdx"].add(pi)
        if score > person["best"][0]:
            person["best"] = (score, j)
        photo_faces.setdefault(pi, []).append({"personId": None, "faceId": face_id,
                                               "bbox": [int(v) for v in bbox], "lab": lab})
        cv2.imwrite(str(faces_dir / f"{face_id}.jpg"), crops[j],
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    # เรียงคนตามจำนวนรูปมาก->น้อย แล้วตั้ง id
    order = sorted(people.items(), key=lambda kv: -len(kv[1]["photoIdx"]))
    lab_to_pid = {}
    people_out = []
    for rank, (lab, pdata) in enumerate(order, start=1):
        pid = f"p{rank:03d}"
        lab_to_pid[lab] = pid
        best_j = pdata["best"][1]
        people_out.append({
            "id": pid,
            "name": f"คนที่ {rank}",
            "cover": f"f{best_j:05d}.jpg",
            "photoCount": len(pdata["photoIdx"]),
            "faceCount": len(pdata["faces"]),
        })

    photos_out = []
    for i, p in enumerate(photos):
        entry = {
            "id": Path(p["name"]).stem,
            "name": p["name"],
            "w": p.get("w"), "h": p.get("h"),
            "faces": [{"personId": lab_to_pid[f["lab"]], "faceId": f["faceId"], "bbox": f["bbox"]}
                      for f in photo_faces.get(i, [])],
        }
        if p.get("fb_drive_id"):
            entry["fbId"] = p["fb_drive_id"]
        else:
            entry["src"] = f"media/{p['name']}"
        if p.get("print_drive_id"):
            entry["printId"] = p["print_drive_id"]
        elif p.get("print_local"):
            entry["printSrc"] = f"media/{p['print_local'].name}"
        photos_out.append(entry)

    data = {
        "project": args.project,
        "slug": slug,
        "generatedAt": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "params": {"eps": args.eps, "minFace": args.min_face, "minScore": args.min_score},
        "people": people_out,
        "photos": photos_out,
    }
    (proj_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # อัปเดต index รวมทุกโปรเจกต์
    idx_path = out_root / "projects" / "index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {"projects": []}
    idx["projects"] = [x for x in idx["projects"] if x["slug"] != slug]
    idx["projects"].append({"slug": slug, "name": args.project,
                            "photoCount": len(photos_out), "peopleCount": len(people_out)})
    idx_path.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")

    log("เสร็จสิ้น ✔")
    log(f"  ข้อมูล: {proj_dir / 'data.json'}")
    log(f"  เปิดดู: {out_root}/index.html?p={slug}")
    log(f"  ลิงก์สำหรับแขก (หลัง deploy): https://<โดเมนคุณ>/index.html?p={slug}")


if __name__ == "__main__":
    main()
