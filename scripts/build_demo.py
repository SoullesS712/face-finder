#!/usr/bin/env python3
"""สร้าง demo data จากรูปตัวอย่าง 11 รูป (Haar detect + manual identity map)
ใช้เฉพาะทำ demo ให้ viewer มีข้อมูลจริงดู — pipeline จริงใช้ process.py + InsightFace"""
import json, shutil
from pathlib import Path
import cv2

SRC = Path("/mnt/user-data/uploads")
OUT = Path("/home/claude/face-finder/viewer/projects/demo-ngan-buat")
MEDIA, FACES = OUT / "media", OUT / "faces"
for d in (MEDIA, FACES): d.mkdir(parents=True, exist_ok=True)

# แผนตัวตนต่อรูป: (จำนวนหน้าใหญ่สุดที่ assign, [ชื่อคนเรียงซ้าย->ขวา])
# คนอื่นๆ ในรูป = singleton อัตโนมัติ
PLAN = {
    "1000069194.jpg": ["พ่อ", "นาค", "แม่", "น้องสาว"],       # ครอบครัว 4 คน แนวนอน
    "1000069195.jpg": ["พ่อ", "นาค", "แม่", "น้องสาว"],       # ครอบครัว 4 คน แนวตั้ง
    "1000069193.jpg": ["นาค"],                                  # นาคเดี่ยว ห่มขาว
    "1000069398.jpg": ["นาค"],                                  # นาคผมยาว
    "1000069401.jpg": ["นาค"],                                  # โกนผม หลับตา
    "1000069399.jpg": ["นาค", "พ่อ"],                           # ใหญ่สุด=นาค รอง=พ่อ(ตัดผม)
    "1000069400.jpg": ["นาค"],                                  # นาค + คนตัดผม(อื่น)
    "1000069402.jpg": ["นาค"],                                  # หัวโล้น+แว่น (ขวา) -> ใหญ่สุดขวา
    "1000069169.jpg": [], "1000069173.jpg": [], "1000069178.jpg": [],  # ภาพหมู่ singleton
}

casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

people = {}   # name -> {faces:[], photos:set(), cover:(area,faceId)}
photos_out, face_n = [], 0

def add_face(name, face_id, photo_id, area):
    p = people.setdefault(name, {"faces": [], "photos": set(), "cover": (0, None)})
    p["faces"].append(face_id); p["photos"].add(photo_id)
    if area > p["cover"][0]: p["cover"] = (area, face_id)

for fname, named in PLAN.items():
    src = SRC / fname
    if not src.exists(): continue
    shutil.copy2(src, MEDIA / fname)
    img = cv2.imread(str(src))
    H, W = img.shape[:2]
    scale = 1200 / max(H, W)
    small = cv2.resize(img, (int(W * scale), int(H * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dets = casc.detectMultiScale(gray, 1.08, 5, minSize=(28, 28))
    boxes = sorted([(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for x, y, w, h in dets],
                   key=lambda b: -(b[2] * b[3]))
    photo_id = Path(fname).stem
    faces = []
    # หน้าใหญ่สุด n อัน -> assign ตามแผน (เรียงซ้าย->ขวาก่อน assign)
    top = sorted(boxes[:len(named)], key=lambda b: b[0]) if named else []
    rest = [b for b in boxes if b not in top]
    if fname == "1000069402.jpg" and len(boxes) >= 2:      # นาคอยู่ขวา
        top = [sorted(boxes[:2], key=lambda b: b[0])[1]]
        rest = [b for b in boxes if b not in top]
    for (x, y, w, h), name in list(zip(top, named)) + [(b, None) for b in rest]:
        face_id = f"f{face_n:05d}"; face_n += 1
        pad = int(.3 * max(w, h))
        crop = img[max(0, y-pad):min(H, y+h+pad), max(0, x-pad):min(W, x+w+pad)]
        ch = int(160 * crop.shape[0] / max(1, crop.shape[1]))
        cv2.imwrite(str(FACES / f"{face_id}.jpg"),
                    cv2.resize(crop, (160, max(1, ch))), [cv2.IMWRITE_JPEG_QUALITY, 88])
        pname = name or f"__single_{face_id}"
        add_face(pname, face_id, photo_id, w * h)
        faces.append({"pname": pname, "faceId": face_id, "bbox": [x, y, x + w, y + h]})
    photos_out.append({"id": photo_id, "name": fname, "w": W, "h": H,
                       "src": f"media/{fname}", "printSrc": f"media/{fname}", "faces": faces})
    print(f"{fname}: {len(boxes)} หน้า (assign {len(top)})")

# จัดอันดับคน
order = sorted(people.items(), key=lambda kv: (-len(kv[1]["photos"]), kv[0]))
pid_map, people_json = {}, []
rank = 1
for name, pd in order:
    pid = f"p{rank:03d}"; pid_map[name] = pid
    display = name if not name.startswith("__single_") else f"คนที่ {rank}"
    people_json.append({"id": pid, "name": display, "cover": pd["cover"][1] + ".jpg",
                        "photoCount": len(pd["photos"]), "faceCount": len(pd["faces"])})
    rank += 1
for ph in photos_out:
    ph["faces"] = [{"personId": pid_map[f["pname"]], "faceId": f["faceId"], "bbox": f["bbox"]} for f in ph["faces"]]

data = {"project": "งานบวช (เดโม่)", "slug": "demo-ngan-buat",
        "generatedAt": "demo", "params": {"engine": "haar-demo"},
        "people": people_json, "photos": photos_out}
(OUT / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
idx = {"projects": [{"slug": "demo-ngan-buat", "name": "งานบวช (เดโม่)",
                     "photoCount": len(photos_out), "peopleCount": len(people_json)}]}
(OUT.parent / "index.json").write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
print(f"\nคน {len(people_json)} | รูป {len(photos_out)} | หน้า {face_n}")
