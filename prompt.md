# Face Finder — Claude Code Pre-Prompt

## บทบาทของคุณ

คุณคือผู้ช่วย dev ที่จะรัน ทดสอบ และปรับปรุงโปรเจกต์ **Face Finder** บนเครื่องนี้ (WSL/Ubuntu) จนใช้งานจริงได้ โค้ดทั้งหมดเขียนเสร็จแล้ว หน้าที่หลักคือรัน spike test → รัน pipeline จริง → ช่วย debug/ปรับจูน → deploy

## โปรเจกต์นี้คืออะไร

เว็บให้แขกในงานอีเวนต์ (งานบวช) ค้นหารูปที่มีตัวเอง:
- ช่างภาพส่งรูปมาเป็น 2 โฟลเดอร์ใน Google Drive: `facebook` (ไฟล์เล็ก) และ `ใหญ่` (ไฟล์ปริ้น) **map กันด้วยชื่อไฟล์**
- Pipeline (`process.py`) ดึงรูปจากโฟลเดอร์ facebook มา detect ใบหน้า + จัดกลุ่มคน แล้ว export เป็น static data
- เว็บ viewer (`viewer/index.html`): เปิดมาแล้ว**แสดงรูปทั้งหมดทันที** แถบใบหน้าอยู่ด้านบน แขกแตะหน้าตัวเองเพื่อกรองเฉพาะรูปที่มีตัวเอง → กดโหลดไฟล์ปริ้นจาก Drive
- ช่างภาพใช้โหมด `?admin=1` แก้ผลจัดกลุ่มได้ 2 ระดับ:
  1. **รวมกลุ่ม (merge)** — เลือกหลายกลุ่มที่เป็นคนเดียวกันจากแถบใบหน้า แล้วกดรวมกลุ่ม
  2. **ย้ายรายหน้า (move)** — เปิดรูปใน lightbox แตะหน้าที่จัดผิดคน แล้วเลือกคนที่ถูกต้อง หรือ "แยกเป็นคนใหม่" (แก้ false merge ได้)
  พร้อมตั้งชื่อคน (แตะชื่อค้าง) และ export data.json ทับของเดิม การแก้ระหว่างทางเก็บเป็น ops log ใน localStorage จนกด export
- รองรับหลายงาน: รัน process.py ต่องาน ได้ slug ต่องาน ลิงก์แขกคือ `index.html?p=<slug>`

## ข้อมูลงานจริงงานแรก

- ชื่องาน: งานบวช (ผู้ใช้จะกำหนดชื่อเอง ถามก่อนรัน)
- รูป 678 รูป (ระบบต้องรองรับได้ถึง 1,000)
- Drive โฟลเดอร์ facebook: `https://drive.google.com/drive/folders/1jP8jTxKH2-pdKMxs7qngFSFGsKkjEBgh`
- Drive โฟลเดอร์ปริ้น: `https://drive.google.com/drive/folders/1PtGX6MRmpUN5E6lx9-MXDPgVwedM7zUL`
- ไม่ต้องทำ PDPA / ระบบขอลบข้อมูล / auth — แขกทุกคนโหลดรูปไหนก็ได้
- **ต้องมี Google Drive API key จากผู้ใช้** — ถ้ายังไม่มี ให้พาไปสร้าง: console.cloud.google.com → enable Drive API → Create API key และเช็คว่าทั้งสองโฟลเดอร์แชร์เป็น "Anyone with the link"

## โครงสร้างโค้ด (มีอยู่แล้ว ห้ามเขียนใหม่จากศูนย์)

```
face-finder/
├── process.py                    # pipeline หลัก (Drive list/download → InsightFace → DBSCAN → export)
├── viewer/index.html             # เว็บทั้งหมดไฟล์เดียว (แขก + admin) — static ล้วน ไม่มี backend
├── viewer/projects/<slug>/       # ผลลัพธ์ต่องาน: data.json + faces/*.jpg (+ media/ ในโหมด local)
├── viewer/projects/index.json    # รายชื่อทุกงาน (process.py อัปเดตให้เอง)
├── spike_test/01_clustering_test.py   # ทดสอบ accuracy → spike1_report.html
├── spike_test/03_filename_mapping.py  # ทดสอบ mapping + สร้าง spike2_embed_test.html
├── scripts/make_standalone.py    # pack งานเป็น HTML ไฟล์เดียว
└── README.md                     # คู่มือฉบับเต็ม อ่านก่อนเริ่ม
```

### สัญญา schema ของ `data.json` (viewer ผูกกับโครงนี้ ห้ามเปลี่ยนโดยไม่แก้ viewer)

```json
{
  "project": "ชื่องาน", "slug": "...",
  "people": [{"id":"p001","name":"คนที่ 1","cover":"f00000.jpg","photoCount":8,"faceCount":10}],
  "photos": [{"id":"IMG_001","name":"IMG_001.jpg","w":1280,"h":720,
              "fbId":"<driveId>","printId":"<driveId>",
              "faces":[{"personId":"p001","faceId":"f00000","bbox":[x1,y1,x2,y2]}]}]
}
```
โหมด local ใช้ `src`/`printSrc` (path ใต้ projects/<slug>/) แทน `fbId`/`printId`

### รูปแบบ URL ที่ viewer ใช้ (อย่าเปลี่ยน)

- thumbnail: `https://drive.google.com/thumbnail?id=<fbId>&sz=w400|w1600`
- ดาวน์โหลดปริ้น: `https://drive.google.com/uc?export=download&id=<printId>`

## ลำดับงาน (ทำตามนี้)

### Step 0 — เตรียมเครื่อง
```bash
pip install insightface onnxruntime opencv-python scikit-learn requests tqdm
```
รันครั้งแรก InsightFace จะโหลดโมเดล buffalo_l (~300MB) เอง ถ้าโหลดโมเดลล้มเหลวให้แจ้ง URL ที่ติดปัญหาและลองใหม่ อย่าเปลี่ยนไปใช้โมเดลอื่นเงียบๆ

### Step 1 — Spike tests (go/no-go)
```bash
cd spike_test
python 03_filename_mapping.py --api-key <KEY> --fb "<ลิงก์โฟลเดอร์ fb>" --pr "<ลิงก์โฟลเดอร์ปริ้น>"
```
- รายงานผล mapping ให้ผู้ใช้: ตรงกันกี่ % ไฟล์ไหนไม่มีคู่
- เปิด `spike2_embed_test.html` ให้ผู้ใช้เช็คว่ารูป Drive แสดงครบ
- spike 1: โหลดรูปตัวอย่าง ~50-100 รูปจาก Drive มาไว้โฟลเดอร์ local แล้ว
  `python 01_clustering_test.py ./sample --eps 0.55` → เปิด `spike1_report.html` ให้ผู้ใช้ตรวจตา
- **เกณฑ์ผ่าน**: mapping >95% / embed ขึ้นครบ / คนหลัก (นาค) แตกไม่เกิน ~3-4 กลุ่ม / ไม่มี false merge เยอะ
- ถ้า false merge เยอะ → ลด eps เป็น 0.5 หรือ 0.45 แล้วรันใหม่ / ถ้าแตกกระจายเกิน → เพิ่มเป็น 0.6

### Step 2 — รัน pipeline จริง
```bash
python process.py --project "<ชื่องานจากผู้ใช้>" \
  --fb-folder "<ลิงก์ fb>" --print-folder "<ลิงก์ปริ้น>" --api-key <KEY>
```
- 678 รูป CPU ~20-45 นาที รูป cache ไว้ที่ `.cache/<slug>/` รันซ้ำไม่โหลดใหม่
- ถ้า RAM/เวลาเป็นปัญหา ลอง `--det-size 960` (เร็วขึ้น แลกหน้าเล็กหลุดบ้าง)

### Step 3 — ตรวจผล + admin
```bash
cd viewer && python -m http.server 8000
```
- เปิด `http://localhost:8000/index.html?p=<slug>&admin=1`
- แนะนำผู้ใช้ตามลำดับ:
  1. ไล่แถบใบหน้า เลือกกลุ่มที่เป็นคนเดียวกัน → **รวมกลุ่ม**
  2. สุ่มเปิดรูปตรวจ ถ้าหน้าไหนจัดผิดคน → แตะหน้านั้น → **ย้ายไปคนที่ถูก** หรือ **แยกเป็นคนใหม่**
  3. ตั้งชื่อคนสำคัญ (แตะชื่อค้าง) → **ส่งออก data.json** → วางทับ `viewer/projects/<slug>/data.json`
- เคสที่ต้องเจอแน่: เจ้าภาพ (นาค) มีทั้งช่วงผมยาว/โกนผม/หัวโล้น/ใส่แว่น จะแตกเป็นหลายกลุ่ม — พฤติกรรมปกติ ใช้รวมกลุ่ม ส่วน false merge (คนละคนโดนจับรวม) ใช้ย้ายรายหน้าแก้ได้โดยไม่ต้องรัน pipeline ใหม่

### Step 4 — Deploy
- `viewer/` เป็น static ล้วน: Vercel / Netlify / GitHub Pages / Cloudflare Pages ได้หมด
- ลิงก์แขก: `https://<domain>/index.html?p=<slug>` — ทดสอบเปิดจากมือถือจริงก่อนส่งแขก
- งานถัดไป = รัน process.py ใหม่ด้วย `--project` ใหม่ ไม่ต้องแตะโค้ด

## กติกาการแก้โค้ด

- แก้ได้เมื่อเจอบั๊กหรือผู้ใช้ขอฟีเจอร์ แต่**รักษา schema data.json และ query param (`?p=`, `?admin=1`) เดิม**
- viewer ต้องเป็น static ไม่มี backend, ไม่เพิ่ม framework/build step — คงเป็น HTML ไฟล์เดียว
- ภาษา UI เป็นไทย โทนสีธีมเดิม (พื้นเข้มอุ่น + ทอง #D9A441)
- ทุกครั้งที่แก้ viewer ให้เช็ค syntax (`node --check` กับ script ที่ extract ออกมา) และเปิดทดสอบผ่าน http.server
- อย่า hardcode API key ลงไฟล์ใดๆ ให้รับผ่าน argument/env เท่านั้น

## ปัญหาที่รู้อยู่แล้ว + วิธีรับมือ

| อาการ | สาเหตุ/ทางแก้ |
|---|---|
| Drive API 403/404 | โฟลเดอร์ไม่ได้แชร์ anyone with link หรือยังไม่ enable Drive API |
| รูป thumbnail ไม่ขึ้นบางรูป | Drive rate limit ชั่วคราว — reload; ถ้าถาวรเช็คสิทธิ์ไฟล์รายตัว |
| หน้าเล็กในภาพหมู่หลุด | ลด `--min-face` เป็น 28 (แลก noise เพิ่ม) หรือเพิ่ม `--det-size 1600` |
| คนเดียวกันแตกหลายกลุ่ม | ปกติสำหรับเคสนาค ใช้ admin merge; หรือเพิ่ม eps ทีละ 0.05 |
| คนละคนโดนรวมกลุ่ม | แก้มือ: เปิดรูป แตะหน้าที่ผิด → ย้าย/แยกคนใหม่ · ถ้าเจอเยอะทั้งชุดค่อยลด eps ทีละ 0.05 แล้วรัน process ใหม่ |
| ดาวน์โหลดไฟล์ปริ้นใหญ่แล้ว Drive เตือน virus scan | พฤติกรรมปกติของไฟล์ >100MB ผ่าน uc?export=download — ไฟล์รูปทั่วไปไม่เจอ |

## นิยาม "เสร็จ"

1. spike ทั้ง 3 ผ่านและรายงานตัวเลขให้ผู้ใช้เห็น
2. งานจริง 678 รูปประมวลผลครบ เปิด viewer เห็นคนถูกจัดกลุ่ม แขก flow ใช้ได้จริงบนมือถือ
3. ผู้ใช้ merge/ย้ายรายหน้า/ตั้งชื่อผ่าน admin ได้ และ export/วางทับ data.json สำเร็จ
4. deploy ขึ้น hosting และเปิดลิงก์ `?p=<slug>` จากมือถือได้

ถามผู้ใช้เมื่อ: ยังไม่มี API key / ตั้งชื่อโปรเจกต์ / เลือก hosting / ผล spike ก้ำกึ่งว่าจะ go หรือปรับจูนต่อ
