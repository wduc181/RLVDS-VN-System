# AGENTS.md

Ghi chú cho agent làm việc trong repo `RLVDS-VN-System`. Mục tiêu là giảm thời gian hỏi lại bối cảnh dự án ở các lần làm việc sau.

## Ngôn ngữ và cách làm việc

- Trao đổi với người dùng bằng tiếng Việt, trừ khi người dùng yêu cầu khác.
- Đây là learning project nhưng code vẫn nên giữ kiểu modular, dễ test và dễ thay component AI/CV.
- Ưu tiên sửa đúng phạm vi yêu cầu. Tránh refactor lớn khi chỉ cần sửa bug nhỏ.
- Khi thay đổi logic xử lý frame/video, phải đọc các test liên quan trước vì repo có nhiều invariant về raw frame, OCR crop và persistence.

## Tổng quan dự án

RLVDS-VN là hệ thống phát hiện xe vượt đèn đỏ tại Việt Nam:

- Input: video/camera góc cố định.
- Detection: YOLOv5 qua `torch.hub.load("ultralytics/yolov5", "custom", ...)`.
- OCR: PaddleOCR là chính, YOLOv5 character OCR là fallback.
- Logic vi phạm hiện tại: đèn đỏ và anchor point của bbox nằm trong polygon vi phạm.
- UI: Streamlit ở `app.py`.
- CLI: `main.py`.
- Storage: SQLite qua repository layer.
- Config: Pydantic settings + YAML + env vars.

## Cấu trúc cần nhớ

- `app.py`: Streamlit app. Có stream loop, quản lý `st.session_state`, spawn OCR microservice, hiển thị video và lưu vi phạm.
- `main.py`: CLI chạy `Pipeline`.
- `config/default.yaml`: cấu hình mặc định.
- `config/settings.py`: Pydantic models, merge `default.yaml`, `config/local.yaml`, env `RLVDS_`.
- `rlvds/core/`: base dataclasses/interfaces, `Pipeline`, `MiniPipeline`, `CachedPipeline`.
- `rlvds/detection/`: YOLOv5 detector, trả về `Detection`.
- `rlvds/ocr/`: PaddleOCR wrapper, OCR server, preprocessing, postprocess, IOU OCR cache.
- `rlvds/spatial/`: polygon zone, point-in-polygon, mask/draw helpers.
- `rlvds/temporal/`: traffic light FSM và violation logic.
- `rlvds/persistence/`: SQLite, Pydantic record models, repository CRUD/image persistence.
- `rlvds/tracking/`: SORT-style tracker. Hiện chưa phải trung tâm của violation flow.
- `training/`: notebook/config training. `training/yolov5/` là code YOLOv5 vendored/upstream, tránh chỉnh nếu yêu cầu không nhắm vào training.
- `docs/ARCHITECTURE.md` và `docs/ProjectOverview.md`: tài liệu dài để đọc khi cần onboarding sâu.

## Cấu hình và dữ liệu local

- `config/local.yaml` được gitignore, dùng cho override local.
- Env override dùng prefix `RLVDS_`, nested bằng `__`, ví dụ `RLVDS_DETECTION__DEVICE=cpu`.
- `get_settings()` có `lru_cache`; trong test nếu đổi env/config cần clear cache.
- `weights/*.pt`, `data/samples/*`, `data/violations/*`, `data/*.db` đều gitignore. Đừng giả định máy khác có model/video/DB.
- Default detection path hiện là `weights/license_plate.pt`.
- Default video path hiện là `data/samples/sample.mp4`.
- Relative paths trong settings thường được resolve theo project root và có thể tự tạo thư mục.

## Lệnh thường dùng

```bash
python -m pytest
python -m pytest tests/test_frame_integrity.py
python -m pytest tests/test_ocr_cache.py tests/test_persistence.py
streamlit run app.py
python main.py --video data/samples/sample.mp4 --no-display
DOCKER_BUILDKIT=0 docker compose up --build
```

Ghi chú:

- Repo chưa có `pyproject.toml` cho formatter/linter ở root.
- Dependency nặng gồm Torch, PaddleOCR, OpenCV. Test unit hiện dùng nhiều fake object, nên thường không cần model thật.
- Docker Compose chạy Streamlit ở `http://localhost:8501`, mount `data/samples` và `weights` read-only, SQLite nằm trong tmpfs `/tmp/rlvds` nên mất khi container bị xóa.

## Invariant kỹ thuật quan trọng

- Frame từ OpenCV là BGR `numpy.ndarray`.
- `Detection.bbox` là `(x1, y1, x2, y2)` theo pixel; anchor point là giữa cạnh dưới bbox `(center_x, y2)`.
- Detection/OCR/persistence phải dùng raw frame hoặc crop từ raw frame. Không dùng frame đã vẽ polygon/bbox/text để OCR hoặc lưu bằng chứng. Xem `tests/test_frame_integrity.py`.
- `ViolationDetector.check_mock_violation()` hiện chỉ kiểm tra đèn đỏ và anchor trong zone. OCR `"unknown"` vẫn có thể là vi phạm để lưu evidence.
- `ViolationRepository.record_violation()` không bỏ evidence khi OCR fail. Nó tạo `plate_text` dạng `OCR_FAILED_<timestamp>_<bbox>` và `status=OCR_FAILED`.
- Một biển số có thể có nhiều record vi phạm; schema hiện đã bỏ UNIQUE trên `plate_text`.
- Khi xóa record, repository chỉ được xóa file nằm dưới `violations_dir`. Không mở rộng logic delete bỏ qua guard này.
- OCR postprocess nằm ở `rlvds/ocr/postprocess.py`; giữ quy tắc validate biển số Việt Nam tập trung ở đây.
- PaddleOCR local bị ép CPU để tránh xung đột cuDNN với PyTorch/CUDA. `app.py` còn spawn OCR microservice CPU ở `127.0.0.1:8502`.
- `CachedPipeline` có thể dùng async OCR; khi dừng app/pipeline phải gọi cleanup/close phù hợp để không để executor/process chạy nền.
- UI Streamlit phụ thuộc nhiều vào `st.session_state`; khi đổi lifecycle Start/Stop phải kiểm tra `_cleanup_video_source()`.

## Quy ước code

- Type hints cho function/method mới.
- Sử dụng `pathlib.Path` cho path mới.
- Dùng `rlvds/utils/logger.py`, hạn chế `print()` trong package chính.
- Tham số tuning như threshold, polygon, FPS, path, cache TTL nên đưa vào `config/settings.py` và `config/default.yaml`, không hardcode trong flow chính.
- SQL nên đi qua `rlvds/persistence/repository.py` hoặc `Database`, không viết trực tiếp trong UI nếu là business logic.
- Giữ UI nhẹ: xử lý ảnh phức tạp nên nằm trong `rlvds/`, không dồn thêm vào `app.py` nếu có thể tách hợp lý.

## Khi đụng từng khu vực nên chạy test nào

- Detection/crop/model loading: `python -m pytest tests/test_detection.py`
- OCR postprocess/recognizer/preprocessor: `python -m pytest tests/test_ocr_pipeline.py tests/test_ocr_recognizer.py`
- OCR cache/CachedPipeline: `python -m pytest tests/test_ocr_cache.py`
- Raw frame, Streamlit helper, persistence crop: `python -m pytest tests/test_frame_integrity.py`
- Polygon/zone: `python -m pytest tests/test_polygon.py`
- Traffic light/violation logic: `python -m pytest tests/test_traffic_light.py`
- SQLite/repository/images/export: `python -m pytest tests/test_persistence.py`

## Các điểm dễ nhầm

- Tài liệu cũ có nhắc tracking/movement như điều kiện violation, nhưng code hiện tại chưa dùng tracking để xác nhận xe đang di chuyển trong main violation flow.
- `app.py` có cả `MiniPipeline` và `CachedPipeline`; khi `ocr_cache.enabled` thì ưu tiên `CachedPipeline`.
- `LicensePlateOCR` thử gọi HTTP OCR microservice trước, rồi mới fallback local PaddleOCR.
- `config/default.yaml` và `DetectionConfig` có thể lệch nếu thêm field mới; cập nhật cả hai khi thêm config user-facing.
- `training/LP_detect.yaml` và `training/LP_ocr.yaml` là training config; runtime model path lấy từ `config/default.yaml`.

## Đề xuất skill có thể thêm sau

Các skill dưới đây nên tạo thành Codex skills riêng nếu dự án cần lặp lại nhiều lần cùng một workflow:

1. `rlvds-cv-pipeline-review`
   - Dùng khi review/sửa detection, OCR, pipeline, raw-frame handling.
   - Checklist: BGR/raw frame, bbox/anchor, crop expand, OCR fail evidence, FPS impact.

2. `rlvds-test-runner`
   - Dùng để chọn đúng subset test theo module thay đổi.
   - Nên biết test nào không cần model thật và cách tránh download dependency nặng.

3. `rlvds-streamlit-runtime`
   - Dùng khi debug `app.py`, Streamlit rerun/session state, Start/Stop lifecycle, OCR microservice port `8502`.
   - Checklist: cleanup video source, DB disconnect, OCR process terminate, cached pipeline close.

4. `rlvds-ocr-vn-plate`
   - Dùng khi sửa nhận diện/format biển số Việt Nam.
   - Nội dung: PaddleOCR result parsing, multi-line plate ordering, confidence threshold, `clean_plate_text`, `format_plate`, `check_valid_plate`.

5. `rlvds-persistence-evidence`
   - Dùng khi sửa SQLite repository hoặc lưu ảnh evidence.
   - Checklist: multiple violations per plate, `OCR_FAILED`, safe delete under `violations_dir`, CSV/stat filters.

6. `rlvds-docker-runtime`
   - Dùng khi đổi Dockerfile/Compose/deploy.
   - Nội dung: CPU Paddle default, GPU override bằng `PADDLE_PACKAGE`, read-only mounted samples/weights, tmpfs DB behavior.

7. `rlvds-yolov5-training`
   - Dùng khi làm việc với `training/`, dataset Roboflow, YOLOv5 notebooks/configs.
   - Cần nhắc rõ `training/yolov5/` là vendored upstream để tránh chỉnh nhầm.
