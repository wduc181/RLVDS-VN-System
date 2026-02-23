"""
RLVDS Processing Pipeline
==========================

Mục đích:
    Orchestrate toàn bộ luồng xử lý — kết nối tất cả modules thành
    pipeline hoàn chỉnh giống camera.py nhưng dạng modular.

Tham chiếu sample code:
    - .github/sample/camera.py (TOÀN BỘ FILE) — đây là pipeline gốc
    - Mỗi phần trong camera.py đã được tách sang module tương ứng

Luồng xử lý (tương ứng camera.py):
    1. Setup     (camera.py L18-41)     → __init__: load model, config, zone, timer
    2. Read      (camera.py L49-51)     → ingestion/video_source.py
    3. Check     (camera.py L53)        → temporal/traffic_light.py
    4. Mask      (camera.py L56-59)     → spatial/polygon.py + zones.py
    5. Detect    (camera.py L60-68)     → detection/detector.py
    6. Crop      (camera.py L69)        → detection/detector.py::crop_plate
    7. Preprocess (camera.py L71)       → ocr/postprocess.py
    8. OCR       (camera.py L74)        → ocr/recognizer.py
    9. Draw      (camera.py L70,77)     → utils/visualization.py
    10. Save     (camera.py L81-88)     → persistence/repository.py
    11. Display  (camera.py L117-124)   → cv2.imshow
    12. Clean    (camera.py L110-113)   → persistence/repository.py::clean_data

===========================================================================
Class cần implement:
===========================================================================

1. Pipeline
   - __init__(config: Settings)
     + Khởi tạo TẤT CẢ components:
       self.video_source = None  # set khi run()
       self.detector = LicensePlateDetector(config.detection.model_path,
                                            config.detection.device)
       self.ocr = LicensePlateOCR(lang=config.ocr.lang,
                                   use_gpu=config.ocr.use_gpu)
       self.zone = ViolationZone(config.spatial.violation_zone)
       self.traffic_light = TrafficLightFSM(
           red_sec=config.temporal.red_duration_sec,
           green_sec=config.temporal.green_duration_sec,
           yellow_sec=config.temporal.yellow_duration_sec
       )
       self.violation_detector = ViolationDetector(self.zone, self.traffic_light)
       self.db = Database(config.database.url)
       self.repo = ViolationRepository(self.db)
       self.config = config

   - run(video_source: str) -> None
     Main loop — tương ứng camera.py dòng 45-127:

     ```
     self.video_source = VideoSource(video_source)
     self.traffic_light.start()
     self.db.connect()
     self.db.create_tables()

     prev_frame_time = 0
     for frame in self.video_source:
         # --- FPS calculation (camera.py L117-121) ---
         new_frame_time = time.time()
         fps = int(1 / (new_frame_time - prev_frame_time)) if (new_frame_time != prev_frame_time) else 0
         prev_frame_time = new_frame_time

         # --- Check light state (camera.py L53) ---
         if self.traffic_light.is_red():
             # ĐÈN ĐỎ → detect violations
             masked = self.zone.apply_mask(frame)
             self.zone.draw(frame)
             detections = self.detector.detect(masked)

             for det in detections:
                 # Crop + preprocess + OCR (camera.py L69-76)
                 crop_img = self.detector.crop_plate(det, frame, expand_ratio=0.15)
                 processed = preprocess_image(crop_img)
                 plate_text = self.ocr.recognize(processed)

                 if plate_text != "unknown":
                     # Draw (camera.py L77)
                     draw_text(frame, plate_text, (det.bbox[0], det.bbox[1]))
                     draw_bbox(frame, det.bbox)

                     # Save violation (camera.py L81-88)
                     violation = self.violation_detector.process_violation(
                         det, frame, plate_text)
                     image_path = self.repo.save_violation_image(frame, ...)
                     self.repo.save(violation)
         else:
             # ĐÈN XANH → clean data (camera.py L110-113)
             self.repo.clean_data()

         # --- Display (camera.py L122-124) ---
         draw_fps(frame, fps)
         cv2.imshow('RLVDS-VN', set_hd_resolution(frame))
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break

     self.stop()
     ```

   - process_frame(frame: np.ndarray) -> list[Violation]
     + Xử lý 1 frame riêng lẻ (cho batch processing hoặc testing)

   - stop() -> None
     + Cleanup:
       self.video_source.release()
       self.db.disconnect()
       cv2.destroyAllWindows()

Dependencies inject qua __init__:
    config: Settings       → từ config/settings.py::get_settings()
    detector               → detection/detector.py
    ocr                    → ocr/recognizer.py
    zone                   → spatial/zones.py
    traffic_light          → temporal/traffic_light.py
    violation_detector     → temporal/violation.py
    db + repo              → persistence/database.py + repository.py

TODO:
    [ ] Import tất cả modules cần thiết
    [ ] Implement class Pipeline
    [ ] Implement __init__ — khởi tạo components từ config
    [ ] Implement run() — main loop tương ứng camera.py
    [ ] Implement process_frame() — xử lý 1 frame
    [ ] Implement stop() — cleanup resources
    [ ] Handle Ctrl+C gracefully (try/except KeyboardInterrupt)
    [ ] Add logging tại mỗi stage (dùng get_logger từ utils/logger.py)
    [ ] Test: chạy pipeline với video → xem output display
"""
