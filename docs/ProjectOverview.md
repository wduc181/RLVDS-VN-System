# Project Overview: RLVDS-VN

RLVDS-VN (Vietnam Red Light Violation Detection System) là hệ thống thị giác máy tính dùng để phát hiện hành vi vượt đèn đỏ và ghi nhận biển số phương tiện trong video/camera góc cố định.

Tài liệu này mô tả project ở mức tổng quan: bài toán, phạm vi, các luồng sử dụng, dữ liệu, thành phần chính và những giới hạn cần hiểu khi vận hành hoặc phát triển tiếp. Chi tiết module và invariant kỹ thuật nằm trong [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Mục tiêu dự án

Mục tiêu của RLVDS-VN là xây dựng một pipeline có thể:

- Đọc video hoặc camera tại một nút giao thông.
- Phát hiện vùng biển số xe trong từng frame.
- Nhận diện ký tự biển số Việt Nam.
- Xác định vi phạm khi phương tiện xuất hiện trong vùng polygon giám sát trong lúc đèn đỏ.
- Lưu lại bằng chứng gồm thông tin biển số, thời gian, trạng thái đèn, ảnh toàn cảnh và ảnh biển số.
- Cung cấp giao diện demo qua Streamlit và CLI để chạy pipeline.

Đây là learning project, nhưng code được tổ chức theo hướng modular để dễ thay model detection, OCR, persistence hoặc logic vi phạm sau này.

## 2. Bài toán

### Input

- Video file trong `data/samples/`.
- Webcam hoặc camera index.
- Luồng camera mạng như RTSP/HTTP nếu OpenCV đọc được.
- Ảnh tĩnh upload trong tab Image OCR của Streamlit.

### Output

- Biển số đã OCR nếu nhận diện hợp lệ.
- Record `OCR_FAILED_*` nếu xe bị xác định vi phạm nhưng OCR không đọc được biển số hợp lệ.
- Trạng thái vi phạm, thời gian, light state, confidence và zone id.
- Ảnh scene có overlay bằng chứng.
- Ảnh plate crop/preprocessed.
- CSV export và thống kê từ SQLite repository.

### Logic vi phạm hiện tại

Một detection được coi là vi phạm khi:

1. Traffic light FSM đang ở trạng thái `RED`.
2. Anchor point của bbox biển số nằm trong polygon vi phạm.

Anchor point là điểm giữa cạnh dưới bbox: `(center_x, y2)`. Đây là điểm gần mặt đường nhất của bbox biển số, phù hợp hơn center point khi so với vạch dừng.

OCR không phải điều kiện bắt buộc để tạo bằng chứng vi phạm. Nếu OCR trả về `"unknown"` hoặc text không hợp lệ, repository vẫn lưu record với `status=OCR_FAILED` để không làm mất evidence.

## 3. Phạm vi hiện tại

Hệ thống tập trung vào bài toán demo/offline với camera góc cố định:

- Phát hiện biển số, không phát hiện toàn bộ thân xe.
- Đèn giao thông được giả lập bằng FSM theo thời gian, chưa đồng bộ sensor/traffic controller thật.
- Polygon vi phạm được cấu hình thủ công trong YAML.
- Tracking đã có module SORT-style nhưng chưa phải điều kiện chính trong violation flow.
- OCR ưu tiên PaddleOCR; YOLOv5 character OCR tồn tại như fallback/extension.

## 4. Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
| --- | --- | --- |
| Ngôn ngữ | Python 3.10 | Runtime chính |
| Video/CV | OpenCV, NumPy | Đọc frame, crop, vẽ overlay, xử lý ảnh |
| Detection | YOLOv5 qua `torch.hub.load` | Phát hiện bbox biển số |
| OCR chính | PaddleOCR | Nhận diện text từ ảnh biển số |
| OCR fallback | YOLOv5 character OCR | Detect từng ký tự khi cần thay thế OCR engine |
| UI | Streamlit | Demo video stream và OCR ảnh |
| Database | SQLite | Lưu record vi phạm |
| Config | Pydantic Settings + YAML | Validate config và override bằng env |
| Tracking | SORT-style tracker | Theo dõi bbox, hiện là module độc lập/optional |
| Test | pytest | Unit tests cho detection, OCR, cache, polygon, persistence |
| Deploy demo | Docker Compose | Chạy Streamlit trong container |

## 5. Luồng sử dụng chính

### Streamlit video stream

`app.py` là giao diện chính:

- Chọn video mẫu từ `data/samples`.
- Bật/tắt detection overlay.
- Chỉnh target FPS và display width.
- Xem FPS, frame index, trạng thái đèn và số vi phạm đã lưu.
- Khi start stream, app có thể spawn OCR microservice CPU ở `127.0.0.1:8502`.
- Khi stop hoặc hết video, app cleanup video source, DB connection, cached pipeline và OCR process.

Streamlit ưu tiên `CachedPipeline` khi `ocr_cache.enabled=true`; nếu tắt cache thì dùng `MiniPipeline`.

### Streamlit Image OCR

Tab Image OCR cho phép upload ảnh `jpg/jpeg/png`, chạy detection và OCR trên ảnh raw, rồi hiển thị bbox, text, confidence và crop biển số.

Luồng này hữu ích để kiểm tra model và OCR mà không cần chạy video stream.

### CLI

`main.py` chạy pipeline bằng terminal:

```bash
python main.py --video data/samples/sample.mp4 --no-display
python main.py --camera 0
```

CLI dùng `Pipeline`, có thể hiển thị OpenCV window nếu không truyền `--no-display`.

### Docker Compose

`docker-compose.yml` chạy Streamlit ở `http://localhost:8501`.

Compose mặc định:

- Mount `./data/samples` vào container ở chế độ read-only.
- Mount `./weights` vào container ở chế độ read-only.
- Đặt SQLite và evidence trong tmpfs `/tmp/rlvds`, phù hợp demo nhưng dữ liệu mất khi container bị xóa.
- Ép detection device và OCR về CPU để dễ chạy trên Docker thường.

## 6. Dữ liệu và model

### Runtime data

Các dữ liệu local không nên commit:

- `weights/*.pt`: model weights.
- `data/samples/*`: video mẫu.
- `data/violations/*`: ảnh evidence.
- `data/*.db`: SQLite database.

Default runtime path:

- Detection model: `weights/license_plate.pt`
- Sample video: `data/samples/sample.mp4`
- SQLite DB: `data/rlvds.db`
- Evidence directory: `data/violations`

### Training assets

Thư mục `training/` chứa notebook và config training:

- `training/LP_detect.yaml`: config YOLO cho license plate detection.
- `training/LP_ocr.yaml`: config YOLO cho character OCR.
- `training/yolov5-plate-detect.ipynb`: notebook detection.
- `training/yolov5-ocr-vn-plate.ipynb`: notebook OCR/character detection.
- `training/yolov5/`: YOLOv5 vendored/upstream, nên tránh sửa nếu không làm việc trực tiếp với training framework.

Dataset được mô tả trong tài liệu cũ là Vietnamese License Plate dataset từ Roboflow. Runtime của project không tự download dataset hoặc weights; người chạy cần đặt model/video tương ứng vào `weights/` và `data/samples/`.

## 7. Cấu hình hệ thống

Config mặc định nằm ở `config/default.yaml`, được validate bằng Pydantic trong `config/settings.py`.

Thứ tự ưu tiên:

```text
RLVDS_* environment variables
config/local.yaml
config/default.yaml
Pydantic defaults
```

Ví dụ override:

```bash
RLVDS_DETECTION__DEVICE=cpu
RLVDS_DETECTION__CONFIDENCE_THRESHOLD=0.6
RLVDS_TEMPORAL__RED_DURATION_SEC=45
RLVDS_DATABASE__URL=sqlite:///data/rlvds.db
```

Các nhóm config chính:

- `video`: source, FPS, resolution, buffer size.
- `detection`: model path, confidence, IoU, image size, device.
- `tracking`: max age, min hits, IoU threshold.
- `spatial`: polygon vi phạm và màu overlay.
- `temporal`: thời lượng đỏ/xanh/vàng và trạng thái ban đầu.
- `ocr`: PaddleOCR config, CPU threads, confidence threshold, enhanced fallback.
- `ocr_cache`: cache IOU, TTL, max size, async OCR, quality frames.
- `preprocessing`: crop expand, upscale, denoise, CLAHE.
- `database`: SQLite URL.
- `paths`: evidence, weights, samples.

## 8. Luồng xử lý tổng quát

```text
Video/Camera frame
    -> giữ raw BGR frame
    -> YOLOv5 detect bbox biển số
    -> crop biển số từ raw frame
    -> OCR qua cache hoặc PaddleOCR
    -> kiểm tra RED + anchor point trong polygon
    -> nếu vi phạm: lưu DB + scene image + plate image
    -> vẽ overlay lên display frame
```

Một nguyên tắc quan trọng là raw frame và display frame được tách riêng. Detection, OCR crop và persistence dùng raw frame; polygon/bbox/text chỉ được vẽ lên display frame hoặc ảnh scene evidence sau khi đã có dữ liệu sạch.

## 9. Detection

`LicensePlateDetector` trong `rlvds/detection/detector.py`:

- Load YOLOv5 custom weights bằng `torch.hub.load("ultralytics/yolov5", "custom", ...)`.
- Nếu file model không tồn tại hoặc load lỗi, detector trả về empty result thay vì crash app.
- Trả về list `Detection` với bbox `(x1, y1, x2, y2)`, confidence, class id và class name.
- `crop_plate()` cắt biển số từ raw frame và mở rộng bbox theo `expand_ratio`.

## 10. OCR

`LicensePlateOCR` trong `rlvds/ocr/recognizer.py` là OCR engine chính.

Luồng nhận diện:

1. Chuẩn bị input nhỏ cho PaddleOCR bằng `prepare_paddle_ocr_input` để upscale/pad crop khi cần.
2. Nếu OCR microservice ở `127.0.0.1:8502` đang chạy, gửi ảnh qua HTTP trước.
3. Nếu microservice không sẵn sàng, fallback về PaddleOCR local trong cùng process.
4. Parse kết quả PaddleOCR, lọc text noise, sort theo vị trí y để hỗ trợ biển 2 dòng.
5. Chuẩn hóa bằng `clean_plate_text`, `format_plate`, `check_valid_plate`.
6. Trả `OCRResult(text, confidence)` hoặc `"unknown"`.

PaddleOCR local bị ép `use_gpu=False` để tránh xung đột cuDNN với PyTorch/CUDA. Config `ocr.use_gpu` có thể tồn tại để giữ interface, nhưng implementation hiện tại ưu tiên CPU cho OCR.

`YOLOv5CharOCR` tồn tại để nhận diện theo từng ký tự, phù hợp thử nghiệm hoặc fallback sau này.

## 11. OCR cache và FPS

PaddleOCR chậm hơn nhiều so với việc reuse kết quả đã đọc. Vì vậy `CachedPipeline` dùng `PlateTrackCache` để match bbox giữa các frame bằng IOU:

- Cache hit: dùng lại text nếu đủ số lần OCR quality.
- Cache miss: crop và chạy OCR.
- TTL theo frame để xóa entry cũ.
- Max size để giới hạn bộ nhớ.
- `ocr_quality_frames` cho phép OCR nhiều lần trên cùng biển số và giữ kết quả confidence tốt hơn.
- Streamlit có thể bật `async_ocr`, frame đầu có thể trả `"unknown"` trong lúc OCR chạy nền.

Khi async OCR trả `"unknown"` tạm thời nhưng điều kiện vi phạm đã đúng, repository vẫn lưu evidence. Các frame sau có thể cập nhật cache bằng text tốt hơn nếu OCR nền hoàn tất.

## 12. Spatial và Temporal

Spatial:

- Polygon vi phạm nằm trong `settings.spatial.violation_zone`.
- `ViolationZone` kiểm tra point-in-polygon bằng `cv2.pointPolygonTest`.
- Hàm mask polygon có sẵn nhưng luồng chính hiện kiểm tra zone sau detection thay vì bắt buộc detect trên frame đã mask.

Temporal:

- `TrafficLightFSM` giả lập chu kỳ `RED -> GREEN -> YELLOW`.
- Dựa trên wall-clock time và modulo theo tổng cycle duration.
- Có thể set `initial_state`.

Violation:

- `ViolationDetector.check_mock_violation()` hiện kiểm tra `is_red()` và `zone.is_in_zone(anchor)`.
- OCR `"unknown"` không làm mất trạng thái vi phạm.
- Việc tránh lưu lặp trong Streamlit được xử lý thêm bằng session cache; repository vẫn cho phép một biển số có nhiều record vi phạm.

## 13. Persistence và evidence

Persistence nằm trong `rlvds/persistence/`:

- `Database`: wrapper SQLite thread-safe với `RLock`, WAL mode, foreign keys và migration schema.
- `ViolationRecord`, `ViolationCreate`, `ViolationUpdate`: Pydantic models cho DB rows.
- `ViolationRepository`: CRUD, image persistence, statistics, export CSV, clean data.

Schema chính là bảng `violations`:

- `plate_text`
- `violation_time`
- `light_state`
- `status`
- `full_image_path`
- `plate_image_path`
- `confidence`
- `zone_id`
- timestamps

Ảnh evidence được lưu dưới:

```text
data/violations/
├── scene/
├── plate/
└── plate_debug/   # chỉ tạo khi debug=true và raw_plate được truyền vào
```

`record_violation()` thực hiện flow nguyên tử:

1. Normalize biển số. Nếu không hợp lệ, tạo `OCR_FAILED_<timestamp>_<bbox>` và `status=OCR_FAILED`.
2. Insert DB row.
3. Lưu scene image và plate image.
4. Update DB paths.
5. Nếu lưu ảnh fail, rollback row và xóa file đã tạo.

Khi delete record, repository chỉ xóa file nằm dưới `violations_dir`; file bên ngoài bị bỏ qua để tránh xóa nhầm.

## 14. Testing và chất lượng

Test suite hiện tập trung vào các invariant cốt lõi:

- Detection load/parse/crop.
- OCR preprocessing, postprocess, PaddleOCR parser, YOLO char OCR.
- OCR cache IOU, TTL, quality frames, async lifecycle.
- Raw-frame integrity trong Pipeline và Streamlit helper.
- Polygon và zone.
- Traffic light FSM.
- SQLite repository, evidence image, OCR_FAILED, CSV/stat filters, safe delete.

Lệnh thường dùng:

```bash
python -m pytest
python -m pytest tests/test_frame_integrity.py
python -m pytest tests/test_ocr_cache.py tests/test_persistence.py
python -m pytest tests/test_ocr_pipeline.py tests/test_ocr_recognizer.py
```

## 15. Giới hạn và hướng mở rộng

Các giới hạn cần hiểu khi đánh giá kết quả:

- Đèn đỏ đang được giả lập theo thời gian, chưa lấy tín hiệu thật từ thiết bị giao thông.
- Polygon cần hiệu chỉnh thủ công cho từng camera/góc quay.
- Detection đang tập trung vào biển số, không xác nhận đầy đủ quỹ đạo thân xe.
- Tracking tồn tại nhưng chưa được dùng làm điều kiện chính để xác nhận xe đang di chuyển.
- OCR dễ bị ảnh hưởng bởi blur, low resolution, glare, góc nghiêng và biển số bị che.
- Async OCR ưu tiên FPS nên kết quả text có thể đến trễ hơn frame vi phạm.
- Dataset, weights và video sample là dữ liệu local; repo không đảm bảo máy khác có sẵn các file này.

Các hướng mở rộng phù hợp:

- Đồng bộ trạng thái đèn từ nguồn thật.
- Tích hợp tracking vào logic vi phạm để phân biệt xe dừng, xe đi qua và xe chỉ lấn vạch.
- UI hiệu chỉnh polygon trực tiếp trên frame.
- Lưu thêm metadata theo camera/lane.
- Tối ưu model và OCR cho biển số 2 dòng, đêm, mưa hoặc camera xa.
- Tách OCR service thành service deploy độc lập thay vì process con của Streamlit.
