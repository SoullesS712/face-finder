# Face Finder — หารูปตัวเองจากงานถ่ายภาพ

ระบบแยกใบหน้าจากรูปงานอีเวนต์ (บวช/แต่ง/รับปริญญา) ให้แขกกดหน้าตัวเองแล้วเห็นทุกรูปที่มีตัวเอง พร้อมปุ่มโหลดไฟล์ปริ้นความละเอียดเต็มจาก Google Drive

## โครงสร้าง

```
face-finder/
├── process.py               # pipeline หลัก: Drive -> detect -> cluster -> export
├── viewer/
│   ├── index.html           # เว็บทั้งหมด (แขก + โหมดช่างภาพ ?admin=1)
│   └── projects/
│       ├── index.json       # รายชื่อทุกงาน (สร้างอัตโนมัติ)
│       └── <slug>/          # ข้อมูลต่องาน: data.json + faces/
├── spike_test/
│   ├── 01_clustering_test.py    # ทดสอบความแม่น clustering + contact sheet
│   └── 03_filename_mapping.py   # ทดสอบ mapping ชื่อไฟล์ + สร้างหน้าเทส embed (spike 2)
├── scripts/
│   ├── build_demo.py        # (ใช้แล้ว) สร้าง demo จากรูปตัวอย่าง
│   └── make_standalone.py   # pack งานหนึ่งเป็น HTML ไฟล์เดียว
└── demo_standalone.html     # เดโม่พร้อมเปิดดูทันที (จากรูปตัวอย่าง 11 รูป)
```

## ติดตั้ง (WSL / Ubuntu)

```bash
pip install insightface onnxruntime opencv-python scikit-learn requests tqdm
```
ครั้งแรกที่รัน InsightFace จะดาวน์โหลดโมเดล buffalo_l (~300MB) อัตโนมัติ

## เตรียม Google Drive API key (ครั้งเดียว ฟรี)

1. https://console.cloud.google.com → สร้างโปรเจกต์
2. APIs & Services → Library → เปิดใช้ **Google Drive API**
3. Credentials → Create credentials → **API key**
4. โฟลเดอร์รูปทั้งสองต้องแชร์เป็น **Anyone with the link (Viewer)**

## ลำดับการใช้งาน

### ขั้น 1 — Spike test (ตัดสิน go/no-go, ~1 ชม.)

```bash
# spike 3 (mapping ชื่อไฟล์) + spike 2 (สร้างหน้าเทส Drive embed)
cd spike_test
python 03_filename_mapping.py --api-key AIza... \
  --fb "https://drive.google.com/drive/folders/1jP8jTxKH2-pdKMxs7qngFSFGsKkjEBgh" \
  --pr "https://drive.google.com/drive/folders/1PtGX6MRmpUN5E6lx9-MXDPgVwedM7zUL"
# -> อ่านผล mapping ใน terminal, เปิด spike2_embed_test.html เช็ครูปขึ้นครบ

# spike 1 (ความแม่น clustering) — ใช้รูปตัวอย่างสัก 50-100 รูปโหลดมาไว้ในเครื่อง
python 01_clustering_test.py ./sample_photos --eps 0.55
# -> เปิด spike1_report.html ตรวจตา
```

เกณฑ์ผ่าน: mapping ตรง >95% / รูป embed ขึ้นครบ / คนหลักแตกไม่เกิน ~3 กลุ่ม

### ขั้น 2 — ประมวลผลจริง

```bash
python process.py --project "งานบวชนัท 2569" \
  --fb-folder "https://drive.google.com/drive/folders/1jP8..." \
  --print-folder "https://drive.google.com/drive/folders/1PtG..." \
  --api-key "AIza..."
```
678 รูปใช้เวลาราว 20-45 นาที (CPU) — รูปถูก cache รันซ้ำเร็วขึ้นมาก

ค่าปรับได้: `--eps 0.5` (เข้มขึ้น รวมกลุ่มยากขึ้น) · `--min-face 48` (ตัดหน้าเล็ก)

### ขั้น 3 — ตรวจ + รวมกลุ่ม (โหมดช่างภาพ)

```bash
cd viewer && python -m http.server 8000
# เปิด http://localhost:8000/index.html?p=<slug>&admin=1
```
- **รวมกลุ่ม**: แตะวงกลมเลือกกลุ่มที่เป็นคนเดียวกัน (เช่น นาคมีผม + นาคหัวโล้น) → กดรวมกลุ่ม
- **ย้ายรายหน้า**: เปิดรูป → แตะหน้าที่จัดผิดคน → เลือกคนที่ถูก หรือแยกเป็นคนใหม่ (ใช้แก้ false merge)
- แตะชื่อค้างเพื่อตั้งชื่อ ("นาคนัท", "คุณแม่")
- เสร็จแล้วกด **ส่งออก data.json** → เอาไฟล์วางทับ `viewer/projects/<slug>/data.json`

### ขั้น 4 — Deploy + ส่งลิงก์ให้แขก

โฟลเดอร์ `viewer/` เป็น static ล้วน วางได้ทุกที่:
```bash
# ง่ายสุด: Vercel
npm i -g vercel && cd viewer && vercel --prod
# หรือ GitHub Pages / Netlify / Cloudflare Pages ก็ได้
```
ลิงก์แขก: `https://<โดเมน>/index.html?p=<slug>` — เปิดมาเห็นรูปทั้งหมดทันที แตะหน้าตัวเองเพื่อกรอง โหลดไฟล์ปริ้นได้เลย
งานใหม่ = รัน process.py ด้วย `--project` ใหม่ → ได้ slug ใหม่ ลิงก์ใหม่ อยู่บนเว็บเดียวกัน

## หมายเหตุ

- รูปไม่ถูกย้ายไปไหน — เว็บ embed thumbnail จาก Drive โดยตรง ปุ่มดาวน์โหลดชี้ไฟล์ปริ้นตัวจริง
- จำกัด 1,000 รูป/งาน ตาม `--max-photos`
- เคสหิน (นาค: มีผม/โกน/แว่น) ระบบจะแตกเป็นหลายกลุ่ม — เป็นเรื่องปกติ ใช้โหมด admin รวมกลุ่มเอา
- ถ้ารูปใน Drive ถูกลบ/เปลี่ยนสิทธิ์ ลิงก์ในเว็บจะเสียตาม
