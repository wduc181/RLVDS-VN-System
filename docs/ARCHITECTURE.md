# RLVDS-VN Architecture

Tài liệu này mô tả kiến trúc kỹ thuật hiện tại của RLVDS-VN sau khi quét lại codebase. Nếu cần góc nhìn sản phẩm, phạm vi và luồng sử dụng, đọc [ProjectOverview.md](ProjectOverview.md).

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cây thư mục](#2-cây-thư-mục)
3. [Invariant dữ liệu](#3-invariant-dữ-liệu)
4. [Entry points](#4-entry-points)
5. [Pipeline layer](#5-pipeline-layer)
6. [Ingestion](#6-ingestion)
7. [Detection](#7-detection)
8. [OCR](#8-ocr)
9. [Spatial và Temporal](#9-spatial-và-temporal)
10. [Persistence](#10-persistence)
11. [Tracking](#11-tracking)
12. [Configuration](#12-configuration)
13. [Visualization và UI](#13-visualization-và-ui)
14. [Docker runtime](#14-docker-runtime)
15. [Testing](#15-testing)
16. [Quyết định thiết kế quan trọng](#16-quyết-định-thiết-kế-quan-trọng)

## 1. Tổng quan kiến trúc

RLVDS-VN được tổ chức theo kiến trúc phân tầng:

```text
app.py / main.py
        |
        v
Pipeline / MiniPipeline / CachedPipeline
        |
        +--> Ingestion: VideoSource, FrameBuffer
        +--> Detection: LicensePlateDetector
        +--> OCR: LicensePlateOCR, OCR server, PlateTrackCache
        +--> Spatial: ViolationZone, polygon utilities
        +--> Temporal: TrafficLightFSM, ViolationDetector
        +--> Persistence: Database, ViolationRepository
        +--> Visualization: overlay, bbox, FPS, light state
```

Luồng xử lý runtime:

```text
OpenCV frame (BGR raw)
    -> detector.detect(raw_frame)
    -> crop plate from raw_frame
    -> OCR direct/cache/async
    -> violation check: RED + anchor point inside polygon
    -> repository.record_violation(raw_frame, crop, metadata)
    -> draw overlay on display frame
```

Điểm quan trọng: frame phục vụ detection/OCR/persistence phải là frame raw. Frame đã vẽ polygon/bbox/text chỉ dùng để hiển thị hoặc làm scene evidence sau khi đã xử lý dữ liệu sạch.

## 2. Cây thư mục

```text
RLVDS-VN-System/
├── app.py                         # Streamlit UI: video stream + image OCR
├── main.py                        # CLI entry point
├── config/
│   ├── default.yaml               # Config mặc định
│   └── settings.py                # Pydantic settings, YAML/env merge
├── rlvds/
│   ├── core/
│   │   ├── base.py                # Dataclasses + abstract interfaces
│   │   ├── pipeline.py            # Full CLI pipeline
│   │   ├── mini_pipeline.py       # Detect -> OCR -> violation
│   │   └── cached_pipeline.py     # MiniPipeline + OCR cache/async
│   ├── ingestion/
│   │   ├── video_source.py        # OpenCV VideoCapture wrapper
│   │   └── frame_buffer.py        # Thread-safe deque buffer
│   ├── detection/
│   │   ├── detector.py            # YOLOv5 detector
│   │   └── models.py              # Detection exports
│   ├── ocr/
│   │   ├── recognizer.py          # PaddleOCR + YOLOv5CharOCR
│   │   ├── ocr_server.py          # CPU OCR HTTP microservice
│   │   ├── preprocessor.py        # Crop/upscale/denoise/CLAHE helpers
│   │   ├── postprocess.py         # VN plate clean/format/validate
│   │   └── plate_cache.py         # IOU-based OCR cache
│   ├── spatial/
│   │   ├── polygon.py             # Mask, draw, point-in-polygon
│   │   └── zones.py               # ViolationZone
│   ├── temporal/
│   │   ├── traffic_light.py       # RED/GREEN/YELLOW FSM
│   │   └── violation.py           # ViolationDetector
│   ├── persistence/
│   │   ├── database.py            # SQLite wrapper + schema
│   │   ├── models.py              # Pydantic DB models
│   │   └── repository.py          # CRUD + image persistence
│   ├── tracking/
│   │   ├── tracker.py             # SORT-style tracker
│   │   ├── track_state.py         # KalmanBoxTracker
│   │   └── bbox_matcher.py        # IOU helper
│   └── utils/
│       ├── logger.py
│       ├── visualization.py
│       └── io.py
├── tests/
├── docs/
├── training/                      # Notebook/config training; yolov5 vendored
├── weights/                       # Local weights, gitignored
└── data/                          # Local samples, DB, evidence, gitignored
```

## 3. Invariant dữ liệu

Các invariant này được test và nên giữ khi sửa logic frame/video:

- Frame từ OpenCV là `numpy.ndarray` BGR.
- `Detection.bbox` có dạng `(x1, y1, x2, y2)` theo pixel.
- Anchor point của detection là giữa cạnh dưới bbox: `(center_x, y2)`.
- Detection, OCR crop và persistence dùng raw frame hoặc crop từ raw frame.
- Không dùng frame đã vẽ overlay để OCR hoặc lưu crop biển số.
- OCR `"unknown"` vẫn có thể là evidence vi phạm nếu điều kiện RED + polygon đúng.
- Repository cho phép nhiều record cho cùng một biển số.
- Khi OCR fail, repository tạo `plate_text` dạng `OCR_FAILED_<timestamp>_<bbox>` và `status=OCR_FAILED`.
- Khi xóa record, repository chỉ xóa file nằm dưới `violations_dir`.

## 4. Entry points

### `app.py`

Streamlit app có hai tab:

- `Video Stream`: chạy video mẫu với overlay, detection/OCR tùy chọn, metrics và persistence.
- `Image OCR`: upload ảnh tĩnh, detect biển số và OCR từng crop.

Lifecycle khi Start video:

1. Cleanup session cũ bằng `_cleanup_video_source()`.
2. Kiểm tra OCR microservice ở `127.0.0.1:8502`.
3. Nếu chưa có server, spawn `rlvds/ocr/ocr_server.py` bằng `subprocess.Popen`.
4. Mở `VideoSource`.
5. Tạo `ViolationZone`, `TrafficLightFSM`, `ViolationDetector`, `FrameBuffer`.
6. Tạo `Database`, `ViolationRepository`, `PlatePreprocessor`.
7. Tạo `LicensePlateDetector` và `LicensePlateOCR`.
8. Chọn `CachedPipeline` nếu `settings.ocr_cache.enabled`, nếu không dùng `MiniPipeline`.
9. Lưu runtime object vào `st.session_state`.

Vòng stream:

```text
read_frame()
copy raw_frame
draw zone on display frame
if detection enabled:
    pipeline.process_frame(raw_frame)
    draw detections on display frame
if result.is_violation:
    crop from raw_frame
    repository.record_violation(...)
draw light/FPS
st.image(display_frame)
```

Khi Stop hoặc hết video, app release video source, disconnect DB, close cached pipeline executor và terminate OCR process nếu app đã spawn process đó.

### `main.py`

CLI parse các tham số:

```bash
python main.py --video data/samples/sample.mp4
python main.py --camera 0
python main.py --video data/samples/sample.mp4 --debug --no-display
```

`main.py` khởi tạo `Pipeline(settings)` và gọi `pipeline.run(source, display=...)`.

Lưu ý: `--config` hiện được parse nhưng chưa được nối vào `get_settings()`. Override thực tế nên dùng `config/local.yaml` hoặc env vars `RLVDS_`.

## 5. Pipeline layer

### `Pipeline`

`rlvds/core/pipeline.py` là orchestration đầy đủ cho CLI:

- Khởi tạo detector, OCR, zone, traffic light, violation detector.
- Khi `_start()` được gọi, mở `VideoSource`, connect SQLite, tạo repository.
- Nếu `ocr_cache.enabled`, tạo `CachedPipeline`; nếu không tạo `MiniPipeline`.
- Loop qua frame, giữ `raw_frame`, tạo `display_frame`, chạy detection theo `inference_interval_frames`, persist violation rồi vẽ overlay.
- `stop()` release video, disconnect DB và destroy OpenCV windows.

`Pipeline` dùng `VideoSource.iter_frames_throttled()` nếu `video.fps > 0`, giúp skip frame bằng `grab()` thay vì decode toàn bộ frame.

### `MiniPipeline`

`rlvds/core/mini_pipeline.py` là flow đơn giản:

```text
detector.detect(frame)
for detection:
    detector.crop_plate(detection, frame)
    ocr.recognize(crop)
    violation_detector.check_mock_violation(plate_text, detection)
```

Output là `MiniPipelineResult(plate_text, detection, is_violation)`.

### `CachedPipeline`

`rlvds/core/cached_pipeline.py` thêm cache OCR:

```text
detector.detect(frame)
for detection:
    cache.match(detection.bbox)
    if miss:
        crop -> OCR -> cache
    if hit and ocr_count < quality_frames:
        OCR thêm lần nữa để cải thiện confidence
    else:
        reuse cached text
    check_mock_violation(...)
cache.cleanup(frame_idx)
```

Các thuộc tính quan trọng:

- `PlateTrackCache`: match bbox theo IOU.
- `ocr_quality_frames`: số lần OCR tối đa trên cùng một cached plate.
- `async_ocr`: khi bật, OCR chạy trong `ThreadPoolExecutor`.

Streamlit truyền `async_ocr=settings.ocr_cache.async_ocr`. `Pipeline` CLI hiện tạo `CachedPipeline` theo flow đồng bộ vì không truyền `async_ocr`.

## 6. Ingestion

### `VideoSource`

`rlvds/ingestion/video_source.py` wrap `cv2.VideoCapture`.

Nguồn hỗ trợ:

- File video: path tồn tại trên disk.
- Webcam: `0`, `1` hoặc string số.
- Stream URL: `rtsp://`, `http://`, `https://`, `rtmp://`, `udp://`.

API chính:

- `read_frame() -> tuple[bool, np.ndarray | None]`
- `grab_frame() -> bool`
- `retrieve_frame()`
- `iter_frames()`
- `iter_frames_throttled(target_fps)`
- `get_fps()`, `get_frame_size()`, `get_frame_count()`
- `release()`

Stream input có cơ chế tolerance lỗi đọc và reconnect. File input dừng ngay khi end-of-stream.

### `FrameBuffer`

`FrameBuffer` là deque thread-safe, có `put`, `get`, `clear`, `is_full` và helper `skip_frames`. Hiện app tạo buffer theo runtime components nhưng luồng xử lý chính vẫn đọc frame tuần tự.

## 7. Detection

### `LicensePlateDetector`

File: `rlvds/detection/detector.py`

Khởi tạo:

```python
LicensePlateDetector(
    model_path=settings.detection.model_path,
    confidence_threshold=settings.detection.confidence_threshold,
    iou_threshold=settings.detection.iou_threshold,
    image_size=settings.detection.image_size,
    device=settings.detection.device,
)
```

Load model:

```python
torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=model_path,
    force_reload=False,
    trust_repo=True,
)
```

Nếu model path không tồn tại hoặc load lỗi:

- `self.model = None`
- `is_available() == False`
- `detect()` trả `[]`

`detect(frame)` parse `results.pandas().xyxy[0]` và tạo `Detection`.

`crop_plate(detection, frame, expand_ratio)`:

- Mở rộng bbox theo phần trăm width/height.
- Clip theo kích thước frame.
- Trả bản copy crop từ raw frame.

## 8. OCR

### `LicensePlateOCR`

File: `rlvds/ocr/recognizer.py`

API:

- `recognize(image) -> str`
- `recognize_with_confidence(image) -> OCRResult`
- `preprocess(image) -> np.ndarray`

Luồng `recognize_with_confidence`:

1. Trả `"unknown"` nếu ảnh rỗng.
2. Tạo input variants bằng `_ocr_input_variants()`:
   - `raw`: `prepare_paddle_ocr_input(image)`
   - `enhanced`: chỉ có khi `enhanced_fallback=True`
3. Nếu `_use_http=True`, thử POST ảnh PNG sang `http://127.0.0.1:8502`.
4. Nếu HTTP fail hoặc không có server, dùng PaddleOCR local nếu engine build thành công.
5. Parse PaddleOCR result bằng `_parse_paddle_result`.
6. Format và validate biển số bằng `_format_valid_result`.
7. Nếu không có result hợp lệ, trả `OCRResult("unknown", 0.0)`.

### OCR microservice

File: `rlvds/ocr/ocr_server.py`

Microservice chạy PaddleOCR CPU trong process riêng:

- Bind `127.0.0.1:8502`.
- Nhận ảnh binary qua HTTP POST.
- Decode bằng OpenCV.
- Chuẩn bị input bằng `prepare_paddle_ocr_input`.
- Trả JSON `{"raw_result": ...}`.

Mục tiêu là tách PaddleOCR khỏi process chính đang dùng PyTorch/YOLO để giảm rủi ro xung đột CUDA/cuDNN và ổn định FPS.

### PaddleOCR CPU policy

`LicensePlateOCR._paddle_kwargs()` hardcode:

```python
"use_gpu": False
```

Vì vậy PaddleOCR local luôn chạy CPU, dù config có `ocr.use_gpu`. Docker cũng mặc định cài Paddle CPU, trừ khi build override `PADDLE_PACKAGE`.

### Preprocessor

File: `rlvds/ocr/preprocessor.py`

`PlatePreprocessor` có các bước:

```text
crop_plate_region -> denoise -> upscale -> CLAHE
```

Trong `run_pipeline(image)`:

1. `denoise()` chuyển grayscale và chạy `fastNlMeansDenoising`.
2. `upscale()` resize bằng `cv2.INTER_CUBIC`.
3. `apply_clahe()` tăng tương phản local.

Preprocessor được dùng cho:

- Enhanced OCR fallback nếu bật.
- Plate image lưu evidence.
- Test OCR preprocessing.

`prepare_paddle_ocr_input()` riêng biệt với `PlatePreprocessor`; nó đảm bảo crop nhỏ được upscale/pad thành ảnh BGR phù hợp hơn cho PaddleOCR.

### Postprocess

File: `rlvds/ocr/postprocess.py`

Các hàm chính:

- `clean_plate_text(raw_text)`
- `format_plate(text)`
- `check_valid_plate(plate)`
- `to_gray(image)`

Quy tắc hiện tại:

- Giữ ký tự `[A-Za-z0-9.-]`, uppercase, bỏ dấu chấm và dấu gạch nối không tin cậy.
- Hai ký tự đầu là mã tỉnh dạng số.
- Series được sửa theo ngữ cảnh alpha/digit.
- Tail phải là 4-5 chữ số.
- Hỗ trợ series đặc biệt hai chữ như `LD`, `NN`, `NG`, `CD`, `KT`, ...
- Từ chối mã tỉnh ngoài `11-99` và một số mã không hợp lệ.

### YOLOv5 character OCR

`YOLOv5CharOCR` detect từng ký tự bằng YOLOv5:

- Nhận 7-10 ký tự.
- Ghép một dòng bằng thứ tự x.
- Nhận diện biển hai dòng bằng kiểm tra độ thẳng hàng tương đối.
- Trả `OCRResult(text, avg_confidence)` hoặc `"unknown"`.

Hiện đây là engine fallback/extension, không phải flow mặc định của Streamlit.

### PlateTrackCache

File: `rlvds/ocr/plate_cache.py`

Entry cache:

```python
CachedPlate(
    plate_text,
    bbox,
    confidence,
    first_seen_frame,
    last_seen_frame,
    ocr_count,
)
```

Matching:

- Tính IOU giữa bbox mới và bbox cached.
- Bỏ qua entry hết TTL.
- Chọn entry IOU cao nhất nếu IOU >= threshold.

Stats:

- `hit_count`
- `miss_count`
- `hit_rate`
- `size`

`add_or_update()` không làm tăng hit/miss stats, tránh sai lệch số liệu cache.

## 9. Spatial và Temporal

### Spatial

`rlvds/spatial/polygon.py`:

- `create_polygon(vertices)`
- `create_mask(frame, polygon)`
- `draw_polygon(frame, polygon)`
- `point_in_polygon(point, polygon)`
- `point_distance_to_polygon(point, polygon)`

`create_polygon([])` trả dummy polygon tại gốc để app không crash khi config rỗng. Polygon ít hơn 3 đỉnh nhưng không rỗng sẽ raise `ValueError`.

`ViolationZone` trong `rlvds/spatial/zones.py`:

- Lưu vertices, zone id, color, thickness.
- `is_in_zone(point)` dùng `point_in_polygon`.
- `apply_mask(frame)` có sẵn nhưng không bắt buộc trong main flow.
- `draw(frame)` vẽ polygon in-place.

### Temporal

`TrafficLightFSM` trong `rlvds/temporal/traffic_light.py`:

```text
RED -> GREEN -> YELLOW -> RED
```

FSM dựa trên wall-clock time:

```python
position = (time.time() - start_time) % cycle_duration
```

API chính:

- `start()`
- `get_state() -> LightState`
- `get_time_remaining()`
- `is_red()`
- `set_state(state)`
- `reset()`

### ViolationDetector

File: `rlvds/temporal/violation.py`

`check_frame(detections)`:

- Reset `det.is_violation`.
- Nếu không đỏ: trả `[]`.
- Nếu đỏ: đánh dấu detection có anchor nằm trong zone.

`check_mock_violation(plate_text, detection)`:

- Lấy trạng thái đèn hiện tại.
- Nếu không đỏ: `False`.
- Nếu anchor trong zone: `True`.
- Không yêu cầu OCR hợp lệ.

`process_violation()` có duplicate set nội bộ và lưu ảnh đơn giản, nhưng luồng persistence chính trong app/CLI hiện dùng `ViolationRepository.record_violation()`.

## 10. Persistence

### Database

File: `rlvds/persistence/database.py`

`Database` là wrapper SQLite:

- `check_same_thread=False`
- `row_factory=sqlite3.Row`
- `RLock` cho thread-safety cơ bản.
- `PRAGMA foreign_keys = ON`
- `PRAGMA journal_mode = WAL`
- `PRAGMA synchronous = NORMAL`

Schema table:

```sql
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_text TEXT NOT NULL,
    violation_time TEXT NOT NULL,
    light_state TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'VIOLATION',
    full_image_path TEXT,
    plate_image_path TEXT,
    confidence REAL NOT NULL DEFAULT 0.0,
    zone_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

Indexes:

- `plate_text`
- `violation_time`
- `light_state`
- `status`
- `zone_id`

Migration:

- Nếu DB cũ có `plate_text TEXT NOT NULL UNIQUE`, table được rebuild để bỏ UNIQUE constraint.
- Lý do: một biển số có thể có nhiều vi phạm.

### Models

File: `rlvds/persistence/models.py`

- `ViolationRecord`: model row DB, cho phép `OCR_FAILED_*`.
- `ViolationCreate`: input strict, yêu cầu biển số hợp lệ.
- `ViolationUpdate`: partial update.
- `ViolationStatistics`, `DailyStat`: dashboard/statistics.

`is_ocr_failed_plate()` nhận diện prefix `OCR_FAILED`.

### Repository

File: `rlvds/persistence/repository.py`

CRUD:

- `save(entity)`
- `create(payload)`
- `get_by_id(id)`
- `get_all(limit, offset)`
- `get_by_plate(plate_text)`
- `get_all_by_plate(plate_text)`
- `get_by_date_range(start, end)`
- `update(id, patch)`
- `update_status(id, status)`
- `delete(id)`
- `count(...)`

Data tools:

- `clean_data()`
- `export_csv(path, filters...)`
- `get_statistics(filters...)`

Evidence flow:

```text
record_violation(...)
    -> normalize plate or build OCR_FAILED id
    -> save DB row
    -> save_violation_images(...)
    -> update row with image paths
    -> rollback row/files if image save fails
```

Image directories:

```text
violations_dir/
├── scene/
├── plate/
└── plate_debug/
```

`save_violation_images()` copies frame before drawing polygon/bbox/text. Plate image ưu tiên `preprocessed_plate`; nếu không có thì crop từ raw frame bằng bbox.

Delete guard:

```text
_safe_remove_file(path)
    -> resolve path
    -> only unlink if path.relative_to(violations_dir) succeeds
```

## 11. Tracking

Tracking nằm trong `rlvds/tracking/` và hiện là module độc lập/optional.

`ObjectTracker` implement SORT-style:

1. Predict bbox mới cho tracks bằng Kalman filter.
2. Tính IOU matrix giữa tracks và detections.
3. Match bằng Hungarian algorithm, fallback greedy nếu thiếu scipy.
4. Update matched tracks.
5. Tạo tracks mới cho detections chưa match.
6. Mark lost cho tracks chưa match.
7. Xóa tracks quá `max_age`.
8. Trả về tracks `CONFIRMED`.

`KalmanBoxTracker` dùng state vector:

```text
[cx, cy, area, aspect_ratio, vx, vy, v_area, v_ar]
```

Tracking chưa được dùng làm điều kiện chính trong `ViolationDetector`. Điều kiện main flow vẫn là RED + anchor point trong polygon.

## 12. Configuration

### Load order

`config/settings.py` định nghĩa `Settings` và `get_settings()`:

```text
config/default.yaml
    -> deep merge config/local.yaml nếu tồn tại
    -> Pydantic Settings
    -> env vars RLVDS_* override
```

Pydantic source order trong code đặt env cao hơn YAML init values.

`get_settings()` có `@lru_cache(maxsize=1)`, nên test hoặc script đổi env/config trong cùng process cần clear cache nếu muốn reload.

### Env override

Nested delimiter là `__`:

```bash
RLVDS_DETECTION__DEVICE=cpu
RLVDS_OCR_CACHE__ASYNC_OCR=false
RLVDS_SPATIAL__VIOLATION_ZONE='[[100,200],[300,200],[300,400],[100,400]]'
```

### Path resolution

- `DatabaseConfig.url` chuyển SQLite path tương đối thành absolute path theo project root.
- `PathsConfig` resolve `violations_dir`, `weights_dir`, `samples_dir` theo project root và tự tạo thư mục.

### Config groups

- `VideoConfig`
- `DetectionConfig`
- `TrackingConfig`
- `SpatialConfig`
- `TemporalConfig`
- `OCRConfig`
- `OCRCacheConfig`
- `PreprocessingConfig`
- `DatabaseConfig`
- `PathsConfig`

Một số field có default trong Pydantic dù không xuất hiện trong `default.yaml`; ví dụ `DetectionConfig.inference_interval_frames`.

## 13. Visualization và UI

`rlvds/utils/visualization.py` cung cấp helper vẽ:

- Detections.
- FPS.
- Light state.
- Zone overlay.
- Resize frame hiển thị.

Quy ước:

- Drawing thay đổi frame in-place.
- App luôn copy raw frame trước khi vẽ.
- Streamlit convert BGR -> RGB trước khi `st.image`.

`draw_detections()` nhận cả result object từ `MiniPipeline`/`CachedPipeline` hoặc `Detection` tùy helper implementation; vì vậy các result cần giữ `.detection`, `.plate_text`, `.is_violation` ổn định.

## 14. Docker runtime

`Dockerfile`:

- Base `python:3.10-slim`.
- Cài system deps cho OpenCV/ffmpeg.
- Cài Paddle CPU mặc định qua build arg `PADDLE_PACKAGE=paddlepaddle==2.6.2`.
- Loại `paddlepaddle-gpu` khỏi requirements Docker tạm thời.
- Cài PyTorch CPU index.
- Chạy `streamlit run app.py`.

`docker-compose.yml`:

```yaml
ports:
  - "8501:8501"
environment:
  RLVDS_DATABASE__URL: sqlite:////tmp/rlvds/rlvds.db
  RLVDS_PATHS__VIOLATIONS_DIR: /tmp/rlvds/violations
  RLVDS_DETECTION__DEVICE: cpu
  RLVDS_OCR__USE_GPU: "false"
tmpfs:
  - /tmp/rlvds:rw,nosuid,nodev,size=512m
volumes:
  - ./data/samples:/app/data/samples:ro
  - ./weights:/app/weights:ro
```

DB/evidence trong compose mặc định là dữ liệu tạm. Nếu cần giữ lại sau khi xóa container, thay tmpfs bằng bind mount hoặc volume.

## 15. Testing

Test suite:

```text
tests/test_detection.py
tests/test_ocr_pipeline.py
tests/test_ocr_recognizer.py
tests/test_ocr_cache.py
tests/test_frame_integrity.py
tests/test_polygon.py
tests/test_traffic_light.py
tests/test_persistence.py
```

Lệnh theo khu vực:

```bash
python -m pytest tests/test_detection.py
python -m pytest tests/test_ocr_pipeline.py tests/test_ocr_recognizer.py
python -m pytest tests/test_ocr_cache.py
python -m pytest tests/test_frame_integrity.py
python -m pytest tests/test_polygon.py
python -m pytest tests/test_traffic_light.py
python -m pytest tests/test_persistence.py
```

Các test unit dùng fake object nhiều, thường không cần model thật. Những test quan trọng khi sửa frame/video:

- `tests/test_frame_integrity.py`: đảm bảo OCR/persistence dùng raw frame.
- `tests/test_ocr_cache.py`: cache IOU, TTL, async OCR.
- `tests/test_persistence.py`: OCR_FAILED, multiple violations per plate, safe delete.

## 16. Quyết định thiết kế quan trọng

### Raw frame tách khỏi display frame

Overlay làm thay đổi pixel. Nếu dùng display frame để OCR hoặc save crop, bbox/text/polygon có thể nhiễu vào biển số. Vì vậy app và pipeline luôn copy raw frame trước khi vẽ.

### OCR fail vẫn lưu evidence

Một xe vượt đèn đỏ vẫn là event cần lưu dù OCR thất bại. Repository biến text không hợp lệ thành `OCR_FAILED_*` thay vì bỏ record.

### PaddleOCR chạy CPU

PyTorch/CUDA và PaddlePaddle GPU dễ xung đột cuDNN. Ảnh biển số nhỏ nên OCR CPU là trade-off ổn định cho demo. Streamlit còn spawn OCR server CPU riêng để cách ly process.

### OCR cache dùng IOU, không dùng tracker chính

Cache chỉ cần biết bbox hiện tại giống bbox đã OCR trước đó hay không. IOU-based cache rẻ hơn tích hợp tracker đầy đủ và đủ tốt cho việc giảm số lần gọi OCR.

### Tracking chưa xác nhận vi phạm

`tracking/` đã có SORT-style tracker, nhưng violation flow hiện không yêu cầu track đang di chuyển. Nếu sau này cần giảm false positive, tracking nên được tích hợp vào `ViolationDetector` hoặc một rule layer mới.

### Repository cho phép nhiều record cùng plate

Schema không UNIQUE trên `plate_text` vì một xe có thể vi phạm nhiều lần. Duplicate control nếu cần nên là logic theo session/event, không phải constraint DB toàn cục.

### `config/local.yaml` là nơi override local

Không nên sửa `config/default.yaml` chỉ để chạy trên máy cá nhân. Dùng `config/local.yaml` hoặc env vars để đổi path, device, polygon và threshold.
