# RLVDS-VN — Hướng dẫn Kiến trúc & Chi tiết Hệ thống

> **Tài liệu dành cho người mới tiếp cận project lần đầu.**
> Đọc hết tài liệu này bạn sẽ hiểu toàn bộ cách hệ thống vận hành, từ tổng quan đến từng dòng code.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Cây thư mục & vai trò từng file](#2-cây-thư-mục--vai-trò-từng-file)
3. [Kiến trúc tổng thể](#3-kiến-trúc-tổng-thể)
4. [Luồng xử lý chi tiết](#4-luồng-xử-lý-chi-tiết)
5. [Tầng Ingestion — Đọc video](#5-tầng-ingestion--đọc-video)
6. [Tầng Detection — Phát hiện biển số](#6-tầng-detection--phát-hiện-biển-số)
7. [Tầng OCR — Nhận diện ký tự](#7-tầng-ocr--nhận-diện-ký-tự)
8. [Tầng Spatial — Vùng không gian](#8-tầng-spatial--vùng-không-gian)
9. [Tầng Temporal — Logic thời gian](#9-tầng-temporal--logic-thời-gian)
10. [Tầng Tracking — Theo dõi đối tượng](#10-tầng-tracking--theo-dõi-đối-tượng)
11. [Tầng Persistence — Lưu trữ](#11-tầng-persistence--lưu-trữ)
12. [Hệ thống Cấu hình](#12-hệ-thống-cấu-hình)
13. [Pipeline & Entry Points](#13-pipeline--entry-points)
14. [Caching & Tối ưu FPS](#14-caching--tối-ưu-fps)
15. [Docker](#15-docker)
16. [Testing](#16-testing)
17. [Phụ lục: Các quyết định thiết kế quan trọng](#17-phụ-lục-các-quyết-định-thiết-kế-quan-trọng)

---

## 1. Tổng quan

**RLVDS-VN** (Red Light Violation Detection System - Vietnam) là hệ thống thị giác máy tính tự động phát hiện hành vi **vượt đèn đỏ** và nhận diện biển số xe tại Việt Nam.

### Bài toán

- **Input:** Video từ camera giám sát cố định tại ngã tư
- **Output:** Danh sách biển số xe vi phạm + thời gian + ảnh bằng chứng
- **Logic vi phạm:** Biển số xe nằm trong vùng polygon giám sát **VÀ** đèn giao thông đang đỏ

### Công nghệ chính

| Lớp | Công nghệ | Vai trò |
|-----|-----------|---------|
| Detection | YOLOv5 (torch.hub) | Tìm vị trí biển số trong frame |
| OCR | PaddleOCR (ppOCRv4) | Đọc ký tự từ ảnh biển số đã crop |
| Xử lý ảnh | OpenCV | Upscale, denoise, CLAHE, vẽ annotation |
| UI | Streamlit | Giao diện web xem video + kết quả |
| Database | SQLite | Lưu trữ vi phạm |
| Config | Pydantic + YAML | Type-safe config, override bằng ENV |
| Tracking | SORT (Kalman + Hungarian) | Theo dõi biển số qua nhiều frame |

---

## 2. Cây thư mục & vai trò từng file

```
RLVDS-VN-System/
│
├── app.py                          # Entry point Streamlit UI
├── main.py                         # Entry point CLI (terminal)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker Compose config
│
├── config/                         # ═══ Hệ thống cấu hình ═══
│   ├── default.yaml                #   Toàn bộ config mặc định
│   ├── settings.py                 #   Pydantic models, load YAML, merge ENV
│   └── __init__.py
│
├── rlvds/                          # ═══ Package chính ═══
│   │
│   ├── core/                       # Lõi kiến trúc
│   │   ├── base.py                 #   Abstract Base Classes + Dataclasses
│   │   ├── pipeline.py             #   Full pipeline (CLI mode)
│   │   ├── mini_pipeline.py        #   Pipeline đơn giản (từng frame)
│   │   └── cached_pipeline.py      #   Pipeline có OCR cache
│   │
│   ├── ingestion/                  # Đọc video
│   │   ├── video_source.py         #   Wrapper cv2.VideoCapture
│   │   └── frame_buffer.py         #   Buffer thread-safe
│   │
│   ├── detection/                  # Phát hiện biển số
│   │   ├── detector.py             #   YOLOv5 detector
│   │   └── models.py               #   Re-export dataclasses
│   │
│   ├── ocr/                        # Nhận diện ký tự
│   │   ├── recognizer.py           #   PaddleOCR + YOLOv5CharOCR
│   │   ├── preprocessor.py         #   Pipeline tiền xử lý ảnh
│   │   ├── postprocess.py          #   Chuẩn hóa text, validate biển số
│   │   └── plate_cache.py          #   OCR cache (IOU matching)
│   │
│   ├── spatial/                    # Logic không gian
│   │   ├── polygon.py              #   Point-in-polygon, mask, draw
│   │   ├── zones.py                #   ViolationZone class
│   │   └── calibration.py          #   Camera calibration (TODO)
│   │
│   ├── temporal/                   # Logic thời gian
│   │   ├── traffic_light.py        #   FSM đèn giao thông
│   │   ├── timing.py               #   Utility thời gian
│   │   └── violation.py            #   ViolationDetector
│   │
│   ├── tracking/                   # Theo dõi đối tượng
│   │   ├── tracker.py              #   SORT implementation
│   │   ├── track_state.py          #   KalmanBoxTracker
│   │   └── bbox_matcher.py         #   Hàm tính IOU
│   │
│   ├── persistence/                # Lưu trữ
│   │   ├── database.py             #   SQLite wrapper
│   │   ├── models.py               #   Pydantic models cho DB
│   │   └── repository.py           #   CRUD + ảnh + export
│   │
│   └── utils/                      # Tiện ích
│       ├── logger.py               #   Logging setup
│       ├── visualization.py        #   Vẽ bbox, text, zone
│       └── io.py                   #   File I/O helpers
│
├── tests/                          # ═══ Test suite ═══
│   ├── test_detection.py
│   ├── test_ocr_pipeline.py
│   ├── test_ocr_recognizer.py
│   ├── test_ocr_cache.py
│   ├── test_polygon.py
│   ├── test_traffic_light.py
│   └── test_persistence.py
│
├── weights/                        # Model weights (.pt)
├── data/                           # Dữ liệu
│   ├── samples/                    #   Video mẫu
│   ├── violations/                 #   Ảnh vi phạm được lưu
│   └── rlvds.db                    #   SQLite database
├── training/                       # Notebooks & configs huấn luyện
├── docs/                           # Tài liệu
└── logs/                           # Log files
```

---

## 3. Kiến trúc tổng thể

Hệ thống được thiết kế theo kiến trúc **phân tầng (layered architecture)** với 7 tầng chính:

```
┌─────────────────────────────────────────────────────────┐
│                    ENTRY POINTS                         │
│         main.py (CLI)          app.py (Streamlit)       │
├─────────────────────────────────────────────────────────┤
│                    PIPELINE LAYER                       │
│    Pipeline  │  MiniPipeline  │  CachedPipeline         │
├─────────────────────────────────────────────────────────┤
│  INGESTION   │  DETECTION  │  OCR       │  PERSISTENCE  │
│  ─────────   │  ─────────  │  ───       │  ───────────  │
│ VideoSource  │  YOLOv5     │ PaddleOCR  │  SQLite       │
│ FrameBuffer  │  Detector   │ Preprocessor│  Repository  │
├──────────────┴─────────────┴────────────┴──────────────┤
│              SPATIAL            │       TEMPORAL        │
│              ───────            │       ────────        │
│         ViolationZone           │   TrafficLightFSM     │
│         Polygon utils           │   ViolationDetector   │
├─────────────────────────────────────────────────────────┤
│                    TRACKING (optional)                  │
│              SORT Tracker + Kalman Filter               │
├─────────────────────────────────────────────────────────┤
│                    CONFIG LAYER                         │
│         default.yaml  →  Pydantic Settings  ←  ENV      │
└─────────────────────────────────────────────────────────┘
```

### Nguyên tắc thiết kế

1. **Interface Segregation:** Mỗi module kế thừa từ Abstract Base Class (trong `core/base.py`). Có thể thay thế implementation dễ dàng (vd: đổi YOLOv5 → YOLOv8, PaddleOCR → Tesseract).

2. **Dependency Injection:** Các component được khởi tạo bên ngoài và inject qua constructor. Pipeline không tự `import` cụ thể detector/OCR nào.

3. **Configuration-driven:** Mọi tham số đều nằm trong `config/default.yaml`, có thể override bằng `config/local.yaml` hoặc biến môi trường.

4. **Protocol-based typing:** Dùng `Protocol` (structural subtyping) thay vì ABC khi chỉ cần duck-typing (vd: `DetectorLike`, `OCRLike`).

---

## 4. Luồng xử lý chi tiết

### 4.1 Tổng quan luồng

```
Video/Camera
    │
    ▼
┌─────────────┐
│ VideoSource │  Đọc từng frame qua OpenCV
└──────┬──────┘
       │ frame (BGR numpy array)
       ▼
┌─────────────────┐
│ YOLOv5 Detector │  Phát hiện bounding box biển số
└──────┬──────────┘
       │ List[Detection] — mỗi Detection có bbox, confidence
       ▼
┌──────────────────────┐
│ Plate OCR Cache      │  Kiểm tra cache: bbox này đã OCR chưa?
│ (IOU-based matching) │
└──────┬───────────────┘
       │
       ├── Cache HIT  → reuse plate_text
       │
       └── Cache MISS → crop ảnh → preprocess → PaddleOCR
              │
              ▼
       ┌──────────────────────┐
       │ PlatePreprocessor    │
       │ upscale 2x → denoise │
       │ → CLAHE contrast     │
       └──────┬───────────────┘
              │ grayscale ảnh đã xử lý
              ▼
       ┌──────────────────────┐
       │ PaddleOCR (ppOCRv4)  │  Nhận diện text
       └──────┬───────────────┘
              │ raw text
              ▼
       ┌──────────────────────┐
       │ Post-process         │  chuẩn hóa → format VN plate
       └──────┬───────────────┘
              │ plate_text chuẩn (vd: "59A-12345")
              ▼
┌──────────────────────┐
│ Violation Check      │
│ ┌──────────────────┐ │
│ │ TrafficLightFSM  │ │  Đèn đang RED?
│ │ ViolationZone    │ │  Anchor point trong zone?
│ └──────────────────┘ │
└──────┬───────────────┘
       │
       ├── VI PHẠM → lưu ảnh + DB
       │
       └── KHÔNG vi phạm → chỉ hiển thị
              │
              ▼
       ┌──────────────────────┐
       │ Visualization        │  Vẽ bbox, text, zone, FPS
       │ Streamlit display    │
       └──────────────────────┘
```

### 4.2 Điều kiện xác định vi phạm

Một detection được coi là vi phạm khi **đồng thời** thỏa mãn:

1. **Đèn giao thông đang ĐỎ** — `TrafficLightFSM.is_red() == True`
2. **Anchor point của biển số nằm trong vùng polygon giám sát** — `ViolationZone.is_in_zone(anchor_point) == True`
3. **OCR đọc được biển số hợp lệ** — không phải `"unknown"` và pass `check_valid_plate()`

> **Anchor point** = điểm giữa cạnh dưới của bounding box `(center_x, y2)`. Đây là điểm gần mặt đường nhất, dùng để xác định xe đã vượt qua vạch dừng hay chưa.

---

## 5. Tầng Ingestion — Đọc video

### File: `rlvds/ingestion/video_source.py`

**Class `VideoSource`** kế thừa `BaseVideoSource`, wrap `cv2.VideoCapture`.

#### Các chế độ nguồn

| Loại | Ví dụ | Cách nhận diện |
|------|-------|---------------|
| File video | `"data/samples/v1.mp4"` | Path tồn tại trên disk |
| Webcam | `0`, `1` | Số nguyên |
| IP Camera | `"rtsp://..."`, `"http://..."` | Prefix protocol |

#### Các method quan trọng

```python
# Đọc 1 frame (decode hoàn chỉnh)
ok, frame = src.read_frame()

# Chỉ grab header (nhanh, ~1ms, không decode) — dùng để skip frame
grabbed = src.grab_frame()

# Đọc toàn bộ frame không giới hạn
for frame in src.iter_frames():
    process(frame)

# Đọc frame ở target FPS — skip frame thừa bằng grab()
for frame in src.iter_frames_throttled(30.0):
    process(frame)
```

#### Cơ chế reconnect cho stream

Khi đọc IP camera (RTSP/HTTP), nếu mất kết nối:
- Đếm `max_read_failures` (mặc định 20)
- Thử reconnect với `max_reconnect_attempts` (mặc định 10)
- `_safe_reopen()` gọi `reopen()` có try/except

#### `iter_frames_throttled` — cơ chế tiết kiệm

Thay vì `read()` tất cả frame rồi bỏ đi (tốn decode), method này:
- Frame cần xử lý → `read_frame()` (decode đầy đủ)
- Frame bỏ qua → `grab_frame()` (chỉ đọc header, không decode)

---

## 6. Tầng Detection — Phát hiện biển số

### File: `rlvds/detection/detector.py`

**Class `LicensePlateDetector`** kế thừa `BaseDetector`.

#### Cách load model

```python
self.model = torch.hub.load(
    "ultralytics/yolov5",    # repo
    "custom",                # custom weights
    path="weights/license_plate.pt",
    force_reload=False,
)
```

#### Cấu hình model

```python
self.model.conf = confidence_threshold  # ngưỡng confidence (default 0.5)
self.model.iou = iou_threshold          # ngưỡng IOU cho NMS (default 0.45)
self.model.to(device)                   # "cuda" hoặc "cpu"
```

#### Method `detect(frame)` → `List[Detection]`

```python
results = self.model(frame, size=640)
# Parse kết quả từ pandas DataFrame
for row in results.pandas().xyxy[0].values.tolist():
    x1, y1, x2, y2, confidence, class_id, class_name = ...
    detections.append(Detection(bbox=(x1,y1,x2,y2), ...))
```

#### Method `crop_plate(detection, frame, expand_ratio=0.15)`

Cắt vùng biển số, mở rộng bbox 15% mỗi phía để tránh mất ký tự ở rìa, clip trong giới hạn frame.

#### `is_available()`

Trả về `True` nếu `self.model is not None` — dùng để fallback khi model file không tồn tại.

---

## 7. Tầng OCR — Nhận diện ký tự

### 7.1 Recognizer — `rlvds/ocr/recognizer.py`

**Class `LicensePlateOCR`** (chính):
- Wrap PaddleOCR với `use_gpu=False` (luôn CPU để tránh xung đột cuDNN giữa PyTorch CUDA 12.4 và PaddlePaddle 2.6.2)
- `recognize(image)` → `str`
- `recognize_with_confidence(image)` → `OCRResult(text, confidence)`

**Class `YOLOv5CharOCR`** (dự phòng):
- Dùng YOLOv5 để detect từng ký tự riêng lẻ
- Tự động phân loại biển 1 dòng (ô tô) vs 2 dòng (xe máy)
- Ghép ký tự theo tọa độ x

#### Flow OCR chính

```
1. PaddleOCR(image, cls=False)
2. Parse kết quả → lọc theo confidence_threshold
3. clean_plate_text() → chuẩn hóa ký tự
4. format_plate() → định dạng biển số VN
```

### 7.2 Preprocessor — `rlvds/ocr/preprocessor.py`

**Pipeline 4 bước**:

```
Crop từ frame → Upscale 2x → Denoise → CLAHE
```

| Bước | Method | Tham số chính |
|------|--------|--------------|
| Crop | `crop_plate_region()` | `expand_ratio=0.15` |
| Upscale | `cv2.INTER_CUBIC` | `upscale_factor=2.0` |
| Denoise | `cv2.fastNlMeansDenoising` | `h=30, template=7, search=21` |
| CLAHE | `cv2.createCLAHE` | `clipLimit=2.0, tileGridSize=(8,8)` |

Kết quả cuối cùng là ảnh **grayscale** đã được tăng cường, sẵn sàng cho OCR.

### 7.3 Post-process — `rlvds/ocr/postprocess.py`

#### `clean_plate_text(raw_text)` → chuỗi đã chuẩn hóa

1. Xóa ký tự không phải `[A-Za-z0-9.-]`
2. Uppercase, bỏ dấu `.`
3. Bỏ tất cả dấu `-` (vì OCR thường đặt sai vị trí)
4. Sửa lỗi OCR phổ biến:
   - Ký tự đầu (province code): `O→0, I→1, Z→2, S→5, G→6, B→8`
   - Ký tự series (chữ cái): `0→O, 1→I, 2→Z, 5→S, 6→G, 8→B`
   - Phần đuôi (số): luôn là digits

#### `format_plate(text)` → định dạng biển số VN

- Biển 7-8 ký tự: `XXX-XXXX` hoặc `XXX-XXXXX`
- Biển 9-10 ký tự (có series 4 ký tự): `XXXX-XXXXX`
- Tự động thêm dấu `-` vào vị trí đúng

#### `check_valid_plate(plate)` → bool

Validate biển số VN:
- Province code: 2 chữ số, `11–99`, không nằm trong danh sách mã tỉnh không hợp lệ
- Prefix: `\d{2}[A-Z]\d?`
- Tail: 4-5 chữ số

---

## 8. Tầng Spatial — Vùng không gian

### File: `rlvds/spatial/polygon.py`

Các hàm utility:

| Hàm | Chức năng |
|-----|----------|
| `create_polygon(vertices)` | List of points → numpy array `(N,1,2)` int32 |
| `create_mask(frame, polygon)` | Tô trắng polygon, phần còn lại đen → bitwise_and |
| `draw_polygon(frame, polygon)` | Vẽ viền polygon lên frame |
| `point_in_polygon(point, polygon)` | `cv2.pointPolygonTest` → True/False |
| `point_distance_to_polygon(point, polygon)` | Khoảng cách có dấu đến polygon |

### File: `rlvds/spatial/zones.py`

**Class `ViolationZone`** kế thừa `BaseSpatialReasoner`:

```python
zone = ViolationZone(
    vertices=[[600,450], [1200,450], [1260,700], [600,700]],
    zone_id="default",
    color=(0, 0, 255),
)

# Kiểm tra điểm trong zone
zone.is_in_zone((cx, y2))

# Vẽ zone lên frame
zone.draw(frame)

# Mask frame (giữ lại phần trong zone)
masked = zone.apply_mask(frame)
```

### Polygon được định nghĩa trong config

```yaml
# config/default.yaml
spatial:
  violation_zone: [[600, 450], [1200, 450], [1260, 700], [600, 700]]
```

Đây là 4 đỉnh của vùng tứ giác giám sát trên mặt đường (vạch dừng đèn đỏ).

---

## 9. Tầng Temporal — Logic thời gian

### 9.1 Traffic Light FSM — `rlvds/temporal/traffic_light.py`

**Class `TrafficLightFSM`** giả lập chu kỳ đèn giao thông.

```
      ┌──────────────────────────────────────────┐
      │  RED (30s) → GREEN (30s) → YELLOW (3s)   │
      │       ↑                         ↓        │
      │       └─────────────────────────┘        │
      └──────────────────────────────────────────┘
```

#### Cơ chế hoạt động

- Dùng **wall-clock time** (không phải frame time)
- `start()` → ghi nhận `start_time`, offset theo `initial_state`
- `get_state()` → tính `elapsed % cycle_duration` → xác định phase hiện tại
- Không cần gọi `update()` thủ công — FSM tự vận hành theo thời gian thực

```python
position = (time.time() - start_time) % cycle_duration

if position < red_end:       return RED      # [0, 30)
elif position < green_end:   return GREEN    # [30, 60)
else:                         return YELLOW   # [60, 63)
```

### 9.2 Violation Detector — `rlvds/temporal/violation.py`

**Class `ViolationDetector`** kết hợp không gian + thời gian.

#### `check_frame(detections)` → List[Detection]

Duyệt từng detection, nếu:
1. Đèn RED
2. Anchor point trong zone

→ Đánh dấu `detection.is_violation = True`

#### `check_mock_violation(plate_text, detection)` → bool

Phiên bản "mock" dùng trong MiniPipeline:
- Thêm logic chống trùng lặp: mỗi chu kỳ đèn đỏ, mỗi biển số chỉ ghi nhận 1 lần
- Tự động clear `recorded_plates` khi đèn chuyển sang GREEN

#### `process_violation(detection, frame, plate_text)` → Violation

- Kiểm tra duplicate
- Lưu ảnh bằng chứng
- Tạo đối tượng `Violation`

---

## 10. Tầng Tracking — Theo dõi đối tượng

### File: `rlvds/tracking/tracker.py`

**Class `ObjectTracker`** implement SORT (Simple Online Realtime Tracking).

#### Thuật toán SORT

```
Với mỗi frame:
  1. Kalman predict vị trí mới cho tất cả track hiện có
  2. Tính IOU matrix giữa tracked bboxes × detection bboxes
  3. Hungarian algorithm → tìm cặp (track, detection) tối ưu
  4. Update track đã match với detection mới
  5. Tạo track mới cho detection không match
  6. Đánh dấu "lost" cho track không match
  7. Xóa track có time_since_update > max_age
```

#### Trạng thái track (lifecycle)

```
TENTATIVE → (đủ min_hits=3) → CONFIRMED
                              → (mất > max_age=30) → DELETED
```

### File: `rlvds/tracking/track_state.py`

**Class `KalmanBoxTracker`**:
- State vector 8 chiều: `[cx, cy, area, aspect_ratio, vx, vy, v_area, v_ar]`
- Constant velocity model
- Dùng thư viện `filterpy` cho Kalman Filter

### File: `rlvds/tracking/bbox_matcher.py`

Hàm `compute_iou(box_a, box_b)` — tính Intersection over Union giữa 2 bounding box. Được dùng chung bởi cả tracker và OCR cache.

---

## 11. Tầng Persistence — Lưu trữ

### 11.1 Database — `rlvds/persistence/database.py`

**Class `Database`** — thin wrapper quanh `sqlite3`:

- Thread-safe với `RLock`
- WAL mode, foreign keys ON
- `check_same_thread=False`
- Migration tự động: loại bỏ UNIQUE constraint trên `plate_text`

### 11.2 Schema

```sql
CREATE TABLE violations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_text      TEXT NOT NULL,
    violation_time  TEXT NOT NULL,        -- ISO 8601
    light_state     TEXT NOT NULL,        -- RED/GREEN/YELLOW
    status          TEXT DEFAULT 'VIOLATION',
    full_image_path TEXT,                 -- ảnh scene
    plate_image_path TEXT,                -- ảnh biển số đã preprocess
    confidence      REAL DEFAULT 0.0,
    zone_id         TEXT DEFAULT 'default',
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_violations_plate_text ON violations(plate_text);
CREATE INDEX idx_violations_time ON violations(violation_time);
CREATE INDEX idx_violations_light_state ON violations(light_state);
CREATE INDEX idx_violations_status ON violations(status);
CREATE INDEX idx_violations_zone ON violations(zone_id);

-- Trigger auto-update updated_at
CREATE TRIGGER trg_violations_updated_at AFTER UPDATE ON violations ...
```

### 11.3 Repository — `rlvds/persistence/repository.py`

**Class `ViolationRepository`** — data access layer đầy đủ:

| Method | Chức năng |
|--------|----------|
| `save(entity)` | Insert violation (có validate plate) |
| `get_by_id(id)` | Lấy 1 record |
| `get_all(limit, offset)` | Lấy danh sách (sorted by time DESC) |
| `get_by_plate(plate)` | Tìm theo biển số |
| `get_by_date_range(start, end)` | Lọc theo khoảng thời gian |
| `update(id, patch)` | Cập nhật partial |
| `delete(id)` | Xóa record + ảnh liên quan |
| `count(**filters)` | Đếm với filter |
| `record_violation(...)` | **Flow nguyên tử**: insert DB trước → lưu ảnh → update paths. Nếu lưu ảnh fail thì rollback DB. |
| `save_violation_images(...)` | Lưu ảnh scene (có vẽ polygon + bbox) và ảnh plate |
| `get_statistics(...)` | Thống kê dashboard |
| `export_csv(path)` | Export ra CSV |
| `clean_data()` | Chuẩn hóa + dedup dữ liệu |

#### Cấu trúc thư mục lưu ảnh

```
data/violations/
├── scene/         # Ảnh toàn cảnh (có vẽ zone + bbox)
├── plate/         # Ảnh biển số đã preprocess
└── plate_debug/   # Ảnh biển số raw (chỉ khi debug=true)
```

#### Record flow (`record_violation`)

```
1. Tạo ViolationRecord → validate
2. INSERT vào DB → lấy violation_id
3. save_violation_images() → lưu scene + plate
4. UPDATE full_image_path, plate_image_path
5. Nếu bước 3-4 lỗi → DELETE row + xóa file → rollback toàn bộ
```

---

## 12. Hệ thống Cấu hình

### File: `config/settings.py`

#### Thứ tự ưu tiên (cao → thấp)

```
Environment Variables (RLVDS_*)
    ↓ ghi đè
config/local.yaml (gitignored)
    ↓ ghi đè
config/default.yaml
    ↓ fallback
Giá trị mặc định trong Pydantic model
```

#### Cách override bằng ENV

```bash
# Cú pháp: RLVDS_{SECTION}__{KEY}=value
RLVDS_DETECTION__CONFIDENCE_THRESHOLD=0.7
RLVDS_DEBUG=true
RLVDS_TEMPORAL__RED_DURATION_SEC=45
RLVDS_SPATIAL__VIOLATION_ZONE='[[100,200],[300,200],[300,400],[100,400]]'
```

#### Cấu trúc Settings

```python
class Settings(BaseSettings):
    video: VideoConfig           # fps, buffer_size, width, height
    detection: DetectionConfig   # model_path, confidence, iou, image_size, device
    tracking: TrackingConfig     # enabled, max_age, min_hits, iou_threshold
    spatial: SpatialConfig       # violation_zone, zone_color
    temporal: TemporalConfig     # red/green/yellow duration, initial_state
    ocr: OCRConfig              # lang, use_gpu, confidence_threshold
    ocr_cache: OCRCacheConfig   # enabled, iou_threshold, max_size, ttl_frames
    preprocessing: PreprocessingConfig  # upscale, denoise, clahe params
    database: DatabaseConfig    # url
    paths: PathsConfig          # violations_dir, weights_dir, samples_dir
    debug: bool
    log_level: str
```

#### Factory function

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # 1. Load default.yaml
    # 2. Merge local.yaml (deep merge)
    # 3. Pydantic tự động override bằng ENV
    # 4. Cache kết quả
```

---

## 13. Pipeline & Entry Points

Hệ thống có **2 entry points** và **3 pipeline implementations**:

### Entry Points

#### `main.py` — CLI mode
```bash
python main.py --video data/samples/v1.mp4
python main.py --camera 0
python main.py --video test.mp4 --debug --no-display
```
- Dùng `Pipeline` (full)
- Có OpenCV window hiển thị
- Nhấn `q` để thoát

#### `app.py` — Streamlit Web UI
```bash
streamlit run app.py
```
- Giao diện web với sidebar điều khiển
- Chọn video, bật/tắt detection, điều chỉnh FPS
- Hiển thị metrics: FPS, frame count, light state, violation count

### Pipeline Implementations

#### `Pipeline` (`core/pipeline.py`) — Full pipeline cho CLI
- Khởi tạo tất cả component
- Loop qua video, xử lý từng frame
- Hiển thị OpenCV window
- Persistence vào SQLite

#### `MiniPipeline` (`core/mini_pipeline.py`) — Pipeline cơ bản
- Flow đơn giản: detect → crop → OCR → violation check
- Không cache
- Mỗi frame xử lý độc lập

#### `CachedPipeline` (`core/cached_pipeline.py`) — Pipeline tối ưu
- **Đây là pipeline chính được dùng trong app.py khi `ocr_cache.enabled=true`**
- YOLO chạy mọi frame
- PaddleOCR chỉ gọi khi cache miss
- Xem chi tiết ở phần Caching bên dưới

---

## 14. Caching & Tối ưu FPS

### Vấn đề

PaddleOCR chậm (~100-200ms mỗi lần gọi). Với video 30 FPS, nếu gọi OCR mỗi frame, FPS thực tế sẽ tụt xuống còn ~5 FPS. Một biển số thường xuất hiện trong nhiều frame liên tiếp → gọi OCR lặp lại là lãng phí.

### Giải pháp: `CachedPipeline` + `PlateTrackCache`

#### Cơ chế

```
Frame N:
  YOLOv5 detect → tìm thấy bbox A
    → cache.match(bbox A)
      ├── HIT:   dùng plate_text từ cache (0ms)
      └── MISS:  crop → preprocess → PaddleOCR → lưu vào cache (100-200ms)
```

#### PlateTrackCache (`ocr/plate_cache.py`)

- **IOU-based matching:** So sánh bbox mới với bbox đã cache. Nếu IOU ≥ threshold → match.
- **TTL:** Mỗi entry tự expire sau `ttl_frames` (default 150)
- **Max size:** Tối đa `max_size` entries (default 50)
- **Quality frames:** Chạy OCR tối đa `ocr_quality_frames` lần (default 3) cho cùng 1 biển số, lấy kết quả có confidence cao nhất

#### Stats

```python
cache.hit_count    # số lần cache hit
cache.miss_count   # số lần cache miss
cache.hit_rate     # tỉ lệ hit (0.0 - 1.0)
cache.size         # số entry hiện tại
```

### Luồng `CachedPipeline._resolve_plate_text()`

```
detection đến
    │
    ▼
cache.match(bbox)
    │
    ├── MISS → OCR → add vào cache → return (text, from_cache=False)
    │
    └── HIT → ocr_count < quality_frames?
              ├── YES → OCR thêm lần nữa → update cache → return (text, False)
              └── NO  → return (cached_text, True)  # skip OCR hoàn toàn
```

---

## 15. Docker

### Dockerfile

- Base image: `python:3.10-slim`
- Cài system deps: `ffmpeg`, `libgl1`, `libgomp1`, etc.
- Mặc định cài **PaddlePaddle CPU** (`paddlepaddle==2.6.2`) để chạy được trên mọi máy
- Build arg `PADDLE_PACKAGE` cho phép chuyển sang GPU

### docker-compose.yml

```yaml
services:
  rlvds:
    build:
      args:
        PADDLE_PACKAGE: paddlepaddle==2.6.2  # hoặc paddlepaddle-gpu==2.6.2
    ports:
      - "8501:8501"
    environment:
      RLVDS_DATABASE__URL: sqlite:////tmp/rlvds/rlvds.db
      RLVDS_DETECTION__DEVICE: cpu
      RLVDS_OCR__USE_GPU: "false"
    tmpfs:
      - /tmp/rlvds  # DB trên RAM disk
    volumes:
      - ./data/samples:/app/data/samples:ro   # read-only
      - ./weights:/app/weights:ro             # read-only
```

#### DB trong tmpfs

SQLite DB được đặt trong `/tmp/rlvds/` (tmpfs) → **mất khi container bị xóa**. Phù hợp cho demo/testing. Muốn persistent thì bỏ `tmpfs` và mount volume.

---

## 16. Testing

### Cấu trúc tests

```
tests/
├── __init__.py
├── test_detection.py        # Test YOLOv5 detector
├── test_ocr_recognizer.py   # Test PaddleOCR engine
├── test_ocr_pipeline.py     # Test MiniPipeline tích hợp
├── test_ocr_cache.py        # Test PlateTrackCache
├── test_polygon.py          # Test point-in-polygon
├── test_traffic_light.py    # Test TrafficLightFSM
├── test_persistence.py      # Test ViolationRepository
└── fixtures/                # Test data (gitignored)
```

### Chạy test

```bash
pytest tests/ -v
pytest tests/test_ocr_cache.py -v    # chạy 1 file
pytest tests/ -v -k "test_cache"     # chạy test theo keyword
```

---

## 17. Phụ lục: Các quyết định thiết kế quan trọng

### 17.1 PaddleOCR luôn chạy trên CPU

```python
# rlvds/ocr/recognizer.py
use_gpu=False  # LUÔN CPU, bất kể config
```

**Lý do:** PyTorch (CUDA 12.4) kéo theo cuDNN 9.x, trong khi PaddlePaddle 2.6.2 chỉ tương thích cuDNN 8.x. Chạy cả 2 trên GPU cùng lúc gây crash. Ảnh biển số nhỏ (~150x50px) nên CPU xử lý đủ nhanh.

### 17.2 OCR Cache IOU threshold thấp (0.3)

**Lý do:** Khi FPS thấp (≤5), displacement của bbox giữa các frame lớn → IOU cao hơn sẽ miss. Threshold 0.3 cân bằng giữa match chính xác và khả năng theo dõi biển số đang di chuyển.

### 17.3 Bỏ UNIQUE constraint trên plate_text

**Lý do:** Một biển số có thể vi phạm nhiều lần (nhiều chu kỳ đèn đỏ khác nhau). Migration tự động rebuild table để bỏ constraint này.

### 17.4 Anchor point = bottom-center của bbox

**Lý do:** Điểm này gần mặt đường nhất, phản ánh chính xác vị trí xe so với vạch dừng. Nếu dùng center point, xe mới chớm vào zone đã bị tính là vi phạm (false positive).

### 17.5 Mock violation check vs Check đầy đủ

`check_mock_violation()` trong MiniPipeline/CachedPipeline kiểm tra nhanh (point-in-zone + is_red). `check_frame()` trong ViolationDetector đầy đủ hơn, có thể mở rộng sau này.

### 17.6 Protocol-based typing thay vì ABC

```python
class DetectorLike(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]: ...
    def crop_plate(self, detection, frame, expand_ratio) -> np.ndarray: ...
```

Dùng `Protocol` cho structural subtyping — không cần kế thừa, chỉ cần object có đúng method signature. Cho phép test với mock dễ dàng hơn.

---

## Tổng kết nhanh — Cách hệ thống vận hành

1. **Config** được load từ YAML → Pydantic validate → ENV override
2. **Pipeline** khởi tạo tất cả component: detector, OCR, zone, traffic light, cache, DB
3. **VideoSource** đọc từng frame từ file/camera
4. **YOLOv5** detect vị trí biển số → trả về `List[Detection]`
5. **CachedPipeline** kiểm tra cache: nếu bbox đã OCR → reuse, nếu chưa → crop + preprocess + PaddleOCR
6. **Post-process** chuẩn hóa text biển số về format VN
7. **ViolationDetector** kiểm tra: đèn đỏ? + anchor trong zone? → `is_violation = True`
8. Nếu vi phạm → **ViolationRepository** lưu ảnh + DB
9. **Visualization** vẽ FPS, bbox, zone overlay, light status lên frame
10. **Streamlit** hiển thị frame đã annotated + metrics real-time
