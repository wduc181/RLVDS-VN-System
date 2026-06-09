"""
RLVDS-VN Streamlit Web Application.

Run:
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
import os
# Force CPU thread limiting to prevent CPU starvation and keep FPS stable
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - production installs streamlit.
    st = None  # type: ignore[assignment]

# Suppress torch deprecation warnings from YOLOv5
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

from config.settings import get_settings
from rlvds.core.mini_pipeline import MiniPipeline
from rlvds.detection import LicensePlateDetector
from rlvds.ingestion import FrameBuffer, VideoSource
from rlvds.ocr.preprocessor import PlatePreprocessor
from rlvds.ocr.recognizer import LicensePlateOCR
from rlvds.persistence import Database, ViolationRepository
from rlvds.spatial import ViolationZone
from rlvds.temporal import TrafficLightFSM, ViolationDetector
from rlvds.tracking import LicensePlateSpeedEstimator
from rlvds.utils.logger import get_logger
from rlvds.utils.visualization import (
    draw_detections,
    draw_fps,
    draw_light_status,
    draw_zone_overlay,
    set_hd_resolution,
)
from rlvds.core.mini_pipeline import MiniPipeline
from rlvds.core.cached_pipeline import CachedPipeline
from rlvds.ocr.plate_cache import PlateTrackCache
from rlvds.detection import LicensePlateDetector
from rlvds.ocr.recognizer import LicensePlateOCR

logger = get_logger(__name__)


@dataclass
class ImageOCRResult:
    detection: Any
    plate_text: str
    ocr_confidence: float
    crop: np.ndarray


def _crop_plate_for_ocr(
    detector: Any,
    detection: Any,
    frame: np.ndarray,
    expand_ratio: float,
) -> np.ndarray:
    if detector is not None and hasattr(detector, "crop_plate"):
        return detector.crop_plate(detection, frame, expand_ratio=expand_ratio)
    return detection.crop(frame)


def _decode_uploaded_image(uploaded_file: Any) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        return None
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def _run_ocr_with_confidence(ocr_engine: Any, crop: np.ndarray) -> tuple[str, float]:
    if crop is None or crop.size == 0 or ocr_engine is None:
        return "unknown", 0.0
    if hasattr(ocr_engine, "recognize_with_confidence"):
        result = ocr_engine.recognize_with_confidence(crop)
        return str(getattr(result, "text", "unknown")), float(
            getattr(result, "confidence", 0.0)
        )
    if hasattr(ocr_engine, "recognize"):
        return str(ocr_engine.recognize(crop)), 0.0
    return "unknown", 0.0


def _process_uploaded_image(
    *,
    raw_image: np.ndarray,
    detector: Any,
    ocr_engine: Any,
    settings: Any,
) -> tuple[np.ndarray, list[ImageOCRResult]]:
    """Detect and OCR plates from an uploaded raw BGR image."""
    display_image = raw_image.copy()
    detections = detector.detect(raw_image) if detector is not None else []
    results: list[ImageOCRResult] = []

    for detection in detections:
        crop = _crop_plate_for_ocr(
            detector,
            detection,
            raw_image,
            settings.preprocessing.expand_ratio,
        )
        plate_text, ocr_confidence = _run_ocr_with_confidence(ocr_engine, crop)
        results.append(
            ImageOCRResult(
                detection=detection,
                plate_text=plate_text,
                ocr_confidence=ocr_confidence,
                crop=crop,
            )
        )

        x1, y1, x2, y2 = detection.bbox
        label = f"{int(detection.confidence * 100)}%"
        if plate_text and plate_text.lower() != "unknown":
            label = f"{plate_text} ({int(ocr_confidence * 100)}%)"
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            display_image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return display_image, results


def _get_image_ocr_components(
    settings: Any,
) -> tuple[LicensePlateDetector | None, LicensePlateOCR | None, bool]:
    detector = st.session_state.get("image_ocr_detector")
    ocr_engine = st.session_state.get("image_ocr_engine")

    if detector is None:
        try:
            detector = LicensePlateDetector(
                model_path=settings.detection.model_path,
                confidence_threshold=settings.detection.confidence_threshold,
                iou_threshold=settings.detection.iou_threshold,
                image_size=settings.detection.image_size,
                device=settings.detection.device,
            )
            st.session_state["image_ocr_detector"] = detector
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize image detector: %s", exc)
            detector = None

    if ocr_engine is None:
        try:
            ocr_engine = LicensePlateOCR(
                lang=settings.ocr.lang,
                use_gpu=settings.ocr.use_gpu,
                confidence_threshold=settings.ocr.confidence_threshold,
                det_model_dir=settings.ocr.det_model_dir,
                rec_model_dir=settings.ocr.rec_model_dir,
                enable_mkldnn=settings.ocr.enable_mkldnn,
                cpu_threads=settings.ocr.cpu_threads,
                use_angle_cls=settings.ocr.use_angle_cls,
                enhanced_fallback=True,
            )
            st.session_state["image_ocr_engine"] = ocr_engine
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize image OCR engine: %s", exc)
            ocr_engine = None

    detection_available = bool(
        detector is not None
        and (
            detector.is_available()
            if hasattr(detector, "is_available")
            else True
        )
    )
    return detector, ocr_engine, detection_available


def _recorded_violation_key(result: Any) -> tuple[str, Any]:
    plate_text = str(getattr(result, "plate_text", "") or "").strip()
    if plate_text and plate_text.lower() != "unknown":
        return ("plate", plate_text)
    return ("ocr_failed", tuple(getattr(result.detection, "bbox", ())))


def _process_stream_frame(
    *,
    frame: np.ndarray,
    zone: ViolationZone,
    traffic_light: TrafficLightFSM,
    settings: Any,
    pipeline: Any | None,
    detector: Any | None,
    repo: ViolationRepository | None,
    preprocessor: PlatePreprocessor | None,
    show_detection: bool,
    detection_available: bool,
    show_zone_overlay: bool,
    show_fps: bool,
    fps: float,
    run_detection: bool = True,
    cached_detection_results: list[Any] | None = None,
) -> tuple[np.ndarray, list[Any], int, str, float]:
    """Process one Streamlit frame while keeping OCR and persistence on raw pixels."""
    will_run_detection = (
        show_detection and run_detection and pipeline is not None and detection_available
    )
    raw_frame = frame.copy() if will_run_detection else frame
    display_frame = frame

    light_state = traffic_light.get_state().value
    time_remaining = traffic_light.get_time_remaining()

    if show_zone_overlay:
        draw_zone_overlay(
            display_frame,
            zone.polygon,
            color=settings.spatial.zone_color,
            alpha=0.25,
            thickness=settings.spatial.zone_thickness,
        )
    else:
        zone.draw(display_frame)

    detection_results: list[Any] = cached_detection_results or []
    did_run_detection = False
    if will_run_detection:
        detection_results = pipeline.process_frame(raw_frame)
        did_run_detection = True

    if show_detection and detection_results:
        draw_detections(display_frame, detection_results)

    saved_violations = 0
    if did_run_detection and repo is not None and detection_results:
        for result in detection_results:
            if not result.is_violation:
                continue
            det = result.detection
            crop = _crop_plate_for_ocr(
                detector,
                det,
                raw_frame,
                settings.preprocessing.expand_ratio,
            )
            processed_plate = None
            if preprocessor is not None and crop.size > 0:
                processed = preprocessor.run_pipeline(crop)
                if processed.size > 0:
                    processed_plate = processed
            inserted_id = repo.record_violation(
                frame=raw_frame,
                detection=det,
                plate_text=result.plate_text,
                light_state=light_state,
                preprocessed_plate=processed_plate,
                raw_plate=crop,
                polygon=zone.polygon,
                zone_id=zone.zone_id,
                confidence=det.confidence,
            )
            if inserted_id is not None:
                saved_violations += 1

    draw_light_status(display_frame, light_state)
    if show_fps:
        draw_fps(display_frame, fps)

    return display_frame, detection_results, saved_violations, light_state, time_remaining


def _list_sample_videos(samples_dir: str) -> list[str]:
    p = Path(samples_dir)
    if not p.is_dir():
        return []
    exts = {".mp4", ".avi", ".mkv", ".mov"}
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts)


def _cleanup_video_source() -> None:
    db = st.session_state.pop("violation_db", None)
    if db is not None:
        try:
            db.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error while disconnecting database: %s", exc)

    src = st.session_state.pop("video_src", None)
    if src is not None:
        src.release()
        logger.info(
            "Video source released after %d frames",
            st.session_state.get("frame_idx", 0),
        )

    cached_pipeline = st.session_state.pop("cached_pipeline", None)
    if cached_pipeline is not None and hasattr(cached_pipeline, "close"):
        try:
            cached_pipeline.close()
        except Exception as exc:
            logger.warning("Error while closing cached pipeline: %s", exc)

    ocr_proc = st.session_state.pop("ocr_server_proc", None)
    if ocr_proc is not None:
        logger.info("Terminating background OCR Microservice process...")
        try:
            ocr_proc.terminate()
            ocr_proc.wait(timeout=2)
        except Exception as exc:
            logger.warning("Error while terminating background OCR process: %s", exc)

    for key in (
        "frame_idx",
        "total_frames",
        "resolution",
        "traffic_light",
        "zone",
        "violation_detector",
        "frame_buffer",
        "violation_count",
        "mini_pipeline",
        "plate_detector",
        "detection_available",
        "violation_repo",
        "plate_preprocessor",
        "recorded_plates_cache",
    ):
        st.session_state.pop(key, None)
    st.session_state.pop("frame_idx", None)
    st.session_state.pop("total_frames", None)
    st.session_state.pop("resolution", None)
    st.session_state.pop("traffic_light", None)
    st.session_state.pop("zone", None)
    st.session_state.pop("violation_detector", None)
    st.session_state.pop("frame_buffer", None)
    st.session_state.pop("violation_count", None)
    st.session_state.pop("mini_pipeline", None)
    st.session_state.pop("detection_available", None)


def _build_runtime_components() -> tuple[ViolationZone, TrafficLightFSM, ViolationDetector, FrameBuffer]:
    settings = get_settings()
    zone = ViolationZone(
        vertices=settings.spatial.violation_zone,
        zone_id="default",
        color=settings.spatial.zone_color,
        thickness=settings.spatial.zone_thickness,
    )
    traffic_light = TrafficLightFSM(
        red_sec=settings.temporal.red_duration_sec,
        green_sec=settings.temporal.green_duration_sec,
        yellow_sec=settings.temporal.yellow_duration_sec,
        initial_state=settings.temporal.initial_state,
    )
    traffic_light.start()
    violation_detector = ViolationDetector(
        zone=zone,
        traffic_light=traffic_light,
        violations_dir=settings.paths.violations_dir,
        zone_id=zone.zone_id,
    )
    frame_buffer = FrameBuffer(max_size=settings.video.buffer_size)
    return zone, traffic_light, violation_detector, frame_buffer


def _build_speed_estimator(settings: Any, fps: float) -> LicensePlateSpeedEstimator | None:
    if not settings.speed.enabled:
        return None
    return LicensePlateSpeedEstimator(
        fps=fps,
        meters_per_pixel=settings.speed.meters_per_pixel,
        speed_limit_kmh=settings.speed.limit_kmh,
        min_track_frames=settings.speed.min_track_frames,
        smoothing_window=settings.speed.smoothing_window,
        iou_threshold=settings.tracking.iou_threshold,
        max_age=settings.tracking.max_age,
        anchor=settings.speed.anchor,
    )


def _render_image_ocr_tab(settings: Any) -> None:
    uploaded_file = st.file_uploader(
        "Upload traffic image",
        type=("jpg", "jpeg", "png"),
        accept_multiple_files=False,
    )
    raw_image = _decode_uploaded_image(uploaded_file)

    if uploaded_file is not None and raw_image is None:
        st.error("Cannot decode uploaded image. Please use a valid JPG or PNG file.")
        return

    if raw_image is None:
        st.info("Upload a traffic image to detect and read license plates.")
        return

    st.image(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), channels="RGB")
    if not st.button("Đọc biển số", type="primary"):
        return

    detector, ocr_engine, detection_available = _get_image_ocr_components(settings)
    if not detection_available or detector is None:
        st.warning("Detection model is not available. Check detection.model_path.")
        return
    if ocr_engine is None:
        st.warning("OCR engine is not available.")
        return

    with st.spinner("Đang phát hiện và đọc biển số..."):
        display_image, results = _process_uploaded_image(
            raw_image=raw_image,
            detector=detector,
            ocr_engine=ocr_engine,
            settings=settings,
        )

    st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), channels="RGB")
    if not results:
        st.info("No license plate detected in the uploaded image.")
        return

    st.dataframe(
        [
            {
                "plate_text": result.plate_text,
                "ocr_confidence": round(result.ocr_confidence, 3),
                "detection_confidence": round(result.detection.confidence, 3),
                "bbox": result.detection.bbox,
            }
            for result in results
        ],
        width="stretch",
        hide_index=True,
    )

    crop_columns = st.columns(min(len(results), 4))
    for idx, result in enumerate(results):
        if result.crop.size == 0:
            continue
        with crop_columns[idx % len(crop_columns)]:
            st.image(
                cv2.cvtColor(result.crop, cv2.COLOR_BGR2RGB),
                channels="RGB",
                caption=result.plate_text,
            )


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required to run app.py")

    st.set_page_config(page_title="RLVDS-VN", layout="wide")
    st.title("RLVDS-VN")

    settings = get_settings()
    video_tab, image_tab = st.tabs(["Video Stream", "Image OCR"])

    with image_tab:
        _render_image_ocr_tab(settings)

    with st.sidebar:
        st.header("Video Source")
        sample_videos = _list_sample_videos(settings.paths.samples_dir)
        if not sample_videos:
            st.warning("No sample video found in data/samples/")
            source_path = None
        else:
            source_path = st.selectbox(
                "Select sample video",
                options=sample_videos,
                index=0,
            )

        display_width = st.slider("Display width (px)", 480, 1920, 1280, step=80)
        show_fps = st.checkbox("Show FPS", value=True)
        show_zone_overlay = st.checkbox("Show zone overlay", value=True)
        show_detection = st.checkbox(
            "Enable plate detection",
            value=False,
            help="Enable detection + OCR overlay",
        )

        if (
            show_detection
            and st.session_state.get("running", False)
            and "mini_pipeline" in st.session_state
            and not st.session_state.get("detection_available", False)
        ):
            st.warning("Detection model is not available. Check detection.model_path.")

        target_fps = st.slider("Target FPS", 1, 60, 30)

        st.divider()
        st.subheader("Spatial Zone")
        if settings.spatial.violation_zone:
            st.caption("Vertices (x, y):")
            st.code(str(settings.spatial.violation_zone), language="python")
        else:
            st.info("spatial.violation_zone is empty. App will use dummy polygon.")

        st.subheader("Traffic Light Cycle")
        st.caption(
            "R/G/Y = "
            f"{settings.temporal.red_duration_sec}/"
            f"{settings.temporal.green_duration_sec}/"
            f"{settings.temporal.yellow_duration_sec} (s)"
        )

        st.subheader("Speed Warning")
        st.caption(
            f"{'Enabled' if settings.speed.enabled else 'Disabled'} | "
            f"limit {settings.speed.limit_kmh:.1f} km/h | "
            f"{settings.speed.meters_per_pixel:.4f} m/px"
        )

        is_running = st.session_state.get("running", False)
        should_start = st.session_state.get("should_start", False)
        effective_running = is_running or should_start
        can_start = source_path is not None and not effective_running

        st.button(
            "Start",
            width="stretch",
            disabled=not can_start,
            on_click=lambda: st.session_state.update(should_start=True),
        )
        st.button(
            "Stop",
            width="stretch",
            disabled=not effective_running,
            on_click=lambda: st.session_state.update(running=False),
        )

    video_placeholder = video_tab.empty()
    metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = (
        video_tab.columns(5)
    )
    fps_display = metrics_col1.empty()
    frame_count_display = metrics_col2.empty()
    resolution_display = metrics_col3.empty()
    light_state_display = metrics_col4.empty()
    timer_display = metrics_col5.empty()
    violation_count_display = video_tab.empty()

    if st.session_state.pop("should_start", False) and source_path:
        _cleanup_video_source()

        # Khởi chạy OCR Microservice độc lập ẩn GPU
        import subprocess
        import sys
        import urllib.request
        import urllib.error

        server_online = False
        try:
            with urllib.request.urlopen("http://127.0.0.1:8502", timeout=0.2) as _:
                server_online = True
        except urllib.error.HTTPError:
            server_online = True
        except Exception:
            server_online = False

        if not server_online:
            logger.info("Starting background OCR Microservice process...")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            proc = subprocess.Popen(
                [sys.executable, "rlvds/ocr/ocr_server.py"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            st.session_state["ocr_server_proc"] = proc

            # Polling check (tối đa 10 giây) để đợi OCR Server khởi động hoàn tất
            status_text = video_tab.empty()
            status_text.info("Đang khởi động OCR Microservice chạy ngầm trên CPU...")
            for _ in range(20):
                time.sleep(0.5)
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8502", timeout=0.2) as _:
                        server_online = True
                        break
                except urllib.error.HTTPError:
                    server_online = True
                    break
                except Exception:
                    pass

            if server_online:
                status_text.success("OCR Microservice đã sẵn sàng!")
                time.sleep(0.5)
                status_text.empty()
            else:
                status_text.warning("Không kết nối được với OCR Microservice. Sẽ tự động dùng CPU cục bộ.")
                time.sleep(1.0)
                status_text.empty()

        try:
            src = VideoSource(source_path)
        except (FileNotFoundError, RuntimeError) as exc:
            video_tab.error(f"Cannot open video source: {exc}")
            return

        w, h = src.get_frame_size()
        total_frames = src.get_frame_count()
        logger.info("Streaming %s - %d frames, %dx%d", source_path, total_frames, w, h)

        st.session_state["video_src"] = src
        st.session_state["frame_idx"] = 0
        st.session_state["total_frames"] = total_frames
        st.session_state["resolution"] = f"{w}x{h}"
        zone, traffic_light, violation_detector, frame_buffer = _build_runtime_components()
        st.session_state["zone"] = zone
        st.session_state["traffic_light"] = traffic_light
        st.session_state["violation_detector"] = violation_detector
        st.session_state["frame_buffer"] = frame_buffer
        st.session_state["violation_count"] = 0
        st.session_state["recorded_plates_cache"] = set()

        try:
            db = Database(settings.database.url)
            repo = ViolationRepository(
                database=db,
                violations_dir=settings.paths.violations_dir,
            )
            preprocessor = PlatePreprocessor(settings.preprocessing)
            st.session_state["violation_db"] = db
            st.session_state["violation_repo"] = repo
            st.session_state["plate_preprocessor"] = preprocessor
            logger.info("Persistence initialized: %s", settings.database.url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize persistence: %s", exc)
            st.session_state["violation_db"] = None
            st.session_state["violation_repo"] = None
            st.session_state["plate_preprocessor"] = None

        try:
            detector = LicensePlateDetector(
                model_path=settings.detection.model_path,
                confidence_threshold=settings.detection.confidence_threshold,
                iou_threshold=settings.detection.iou_threshold,
                image_size=settings.detection.image_size,
                device=settings.detection.device,
            )
            ocr_engine = LicensePlateOCR(
                lang=settings.ocr.lang,
                use_gpu=settings.ocr.use_gpu,
                confidence_threshold=settings.ocr.confidence_threshold,
                det_model_dir=settings.ocr.det_model_dir,
                rec_model_dir=settings.ocr.rec_model_dir,
                enable_mkldnn=settings.ocr.enable_mkldnn,
                cpu_threads=settings.ocr.cpu_threads,
                use_angle_cls=settings.ocr.use_angle_cls,
                enhanced_fallback=settings.ocr.enhanced_fallback,
            )
            speed_estimator = _build_speed_estimator(settings, fps=float(target_fps))

            # Chọn pipeline: CachedPipeline (tối ưu FPS) hoặc MiniPipeline (gốc)
            if settings.ocr_cache.enabled:
                plate_cache = PlateTrackCache(
                    iou_threshold=settings.ocr_cache.iou_threshold,
                    max_size=settings.ocr_cache.max_cache_size,
                    ttl_frames=settings.ocr_cache.cache_ttl_frames,
                )
                pipeline = CachedPipeline(
                    detector=detector,
                    ocr=ocr_engine,
                    violation_detector=violation_detector,
                    cache=plate_cache,
                    crop_expand_ratio=settings.preprocessing.expand_ratio,
                    ocr_quality_frames=settings.ocr_cache.ocr_quality_frames,
                    async_ocr=settings.ocr_cache.async_ocr,
                    speed_estimator=speed_estimator,
                )
                st.session_state["cached_pipeline"] = pipeline
                logger.info("CachedPipeline initialized (iou_thresh=%.2f, ttl=%d, async_ocr=%s)",
                            settings.ocr_cache.iou_threshold,
                            settings.ocr_cache.cache_ttl_frames,
                            settings.ocr_cache.async_ocr)
            else:
                pipeline = MiniPipeline(
                    detector=detector,
                    ocr=ocr_engine,
                    violation_detector=violation_detector,
                    crop_expand_ratio=settings.preprocessing.expand_ratio,
                    speed_estimator=speed_estimator,
                )
                st.session_state["mini_pipeline"] = pipeline
                logger.info("MiniPipeline initialized (cache disabled)")

            st.session_state["detection_available"] = detector.is_available()
            st.session_state["plate_detector"] = detector
            if detector.is_available():
                logger.info("Detection pipeline initialized successfully")
            else:
                logger.warning("Detection model not available - detection disabled")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize detection pipeline: %s", exc)
            st.session_state["mini_pipeline"] = None
            st.session_state["cached_pipeline"] = None
            st.session_state["plate_detector"] = None
            st.session_state["detection_available"] = False

        st.session_state["running"] = True

    if not st.session_state.get("running", False):
        if "video_src" in st.session_state:
            _cleanup_video_source()
        video_placeholder.info("Press Start to begin video stream.")
        return

    src = st.session_state.get("video_src")
    zone = st.session_state.get("zone")
    traffic_light = st.session_state.get("traffic_light")
    if src is None or not src.is_opened():
        _cleanup_video_source()
        st.session_state["running"] = False
        video_placeholder.warning("Video source is not available.")
        return
    if zone is None or traffic_light is None:
        _cleanup_video_source()
        st.session_state["running"] = False
        video_placeholder.warning("Spatial/Temporal components are not initialized.")
        return

    total_frames = st.session_state.get("total_frames", 0)
    resolution_display.metric("Resolution", st.session_state.get("resolution", "-"))

    frame_interval = 1.0 / target_fps
    prev_time = time.perf_counter()

    while st.session_state.get("running", False):
        ok, frame = src.read_frame()
        if not ok or frame is None:
            frame_idx = st.session_state.get("frame_idx", 0)
            _cleanup_video_source()
            st.session_state["running"] = False
            video_placeholder.success(f"Completed - processed {frame_idx} frames.")
            break
        # Always copy the raw frame for clean detection and OCR crops.
        # This prevents zone overlays and bounding boxes from polluting the OCR input.
        raw_frame = frame.copy()

        now = time.perf_counter()
        dt = now - prev_time
        fps = int(1 / dt) if dt > 0 else 0
        prev_time = now

        frame_idx = st.session_state.get("frame_idx", 0) + 1
        st.session_state["frame_idx"] = frame_idx

        light_state = traffic_light.get_state().value
        time_remaining = traffic_light.get_time_remaining()

        if show_zone_overlay:
            draw_zone_overlay(
                frame,
                zone.polygon,
                color=settings.spatial.zone_color,
                alpha=0.25,
                thickness=settings.spatial.zone_thickness,
            )
        else:
            zone.draw(frame)

        detection_results = []
        if show_detection:
            # Ưu tiên CachedPipeline, fallback sang MiniPipeline
            pipeline = (
                st.session_state.get("cached_pipeline")
                or st.session_state.get("mini_pipeline")
            )
            if pipeline and st.session_state.get("detection_available", False):
                try:
                    detection_results = pipeline.process_frame(
                        raw_frame,
                        frame_idx=frame_idx,
                        fps=float(target_fps),
                    )
                    draw_detections(frame, detection_results)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Detection failed on frame %d: %s", frame_idx, exc)

        saved_violations = 0
        repo = st.session_state.get("violation_repo")
        preprocessor = st.session_state.get("plate_preprocessor")
        detector = st.session_state.get("plate_detector")
        recorded_cache = st.session_state.setdefault("recorded_plates_cache", set())

        if repo is not None and detection_results:
            for result in detection_results:
                if not result.is_violation:
                    continue

                # Bỏ qua nếu vi phạm này đã được ghi nhận trong phiên chạy hiện tại.
                cache_key = _recorded_violation_key(result)
                if cache_key in recorded_cache:
                    continue

                det = result.detection
                crop = _crop_plate_for_ocr(
                    detector,
                    det,
                    raw_frame,
                    settings.preprocessing.expand_ratio,
                )
                processed_plate = None
                if preprocessor is not None and crop.size > 0:
                    processed = preprocessor.run_pipeline(crop)
                    if processed.size > 0:
                        processed_plate = processed
                inserted_id = repo.record_violation(
                    frame=raw_frame,
                    detection=det,
                    plate_text=result.plate_text,
                    light_state=light_state,
                    preprocessed_plate=processed_plate,
                    raw_plate=crop,
                    polygon=zone.polygon,
                    zone_id=zone.zone_id,
                    confidence=det.confidence,
                )
                if inserted_id is not None:
                    # Thêm vào cache để tránh xử lý lặp lại ở các frame tiếp theo
                    recorded_cache.add(cache_key)
                    saved_violations += 1

        if saved_violations > 0:
            current_count = st.session_state.get("violation_count", 0)
            st.session_state["violation_count"] = current_count + saved_violations

        draw_light_status(frame, light_state)
        if show_fps:
            draw_fps(frame, fps)

        display_frame = set_hd_resolution(frame, width=display_width)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        try:
            video_placeholder.image(display_frame, channels="RGB")
        except Exception:  # noqa: BLE001
            pass
        fps_display.metric("FPS", fps)
        frame_count_display.metric("Frame", f"{frame_idx}/{total_frames}")
        light_state_display.metric("Light State", light_state)
        timer_display.metric("Time Remaining (s)", f"{time_remaining:.1f}")
        violation_count_display.metric(
            "Violation Count",
            st.session_state.get("violation_count", 0),
        )

        elapsed = time.perf_counter() - now
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
