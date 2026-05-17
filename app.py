"""
RLVDS-VN Streamlit Web Application.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import cv2
import streamlit as st

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
from rlvds.utils.logger import get_logger
from rlvds.utils.visualization import (
    draw_detections,
    draw_fps,
    draw_light_status,
    draw_zone_overlay,
    set_hd_resolution,
)
from rlvds.core.cached_pipeline import CachedPipeline
from rlvds.ocr.plate_cache import PlateTrackCache

logger = get_logger(__name__)


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
        "cached_pipeline",
        "detection_available",
        "violation_repo",
        "plate_preprocessor",
    ):
        st.session_state.pop(key, None)


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


def main() -> None:
    st.set_page_config(page_title="RLVDS-VN", layout="wide")
    st.title("RLVDS-VN - Video Stream Test")

    settings = get_settings()

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

        is_running = st.session_state.get("running", False)
        should_start = st.session_state.get("should_start", False)
        effective_running = is_running or should_start
        can_start = source_path is not None and not effective_running

        st.button(
            "Start",
            use_container_width=True,
            disabled=not can_start,
            on_click=lambda: st.session_state.update(should_start=True),
        )
        st.button(
            "Stop",
            use_container_width=True,
            disabled=not effective_running,
            on_click=lambda: st.session_state.update(running=False),
        )

    video_placeholder = st.empty()
    metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
    fps_display = metrics_col1.empty()
    frame_count_display = metrics_col2.empty()
    resolution_display = metrics_col3.empty()
    light_state_display = metrics_col4.empty()
    timer_display = metrics_col5.empty()
    violation_count_display = st.empty()

    if st.session_state.pop("should_start", False) and source_path:
        _cleanup_video_source()

        try:
            src = VideoSource(source_path)
        except (FileNotFoundError, RuntimeError) as exc:
            st.error(f"Cannot open video source: {exc}")
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
            )

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
                )
                st.session_state["cached_pipeline"] = pipeline
                logger.info("CachedPipeline initialized (iou_thresh=%.2f, ttl=%d)",
                            settings.ocr_cache.iou_threshold,
                            settings.ocr_cache.cache_ttl_frames)
            else:
                pipeline = MiniPipeline(
                    detector=detector,
                    ocr=ocr_engine,
                    violation_detector=violation_detector,
                    crop_expand_ratio=settings.preprocessing.expand_ratio,
                )
                st.session_state["mini_pipeline"] = pipeline
                logger.info("MiniPipeline initialized (cache disabled)")

            st.session_state["detection_available"] = detector.is_available()
            if detector.is_available():
                logger.info("Detection pipeline initialized successfully")
            else:
                logger.warning("Detection model not available - detection disabled")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize detection pipeline: %s", exc)
            st.session_state["mini_pipeline"] = None
            st.session_state["cached_pipeline"] = None
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
        now = time.perf_counter()

        if now - prev_time < frame_interval:
            # Skip — grab header only (fast, ~1ms, no decode)
            if not src.grab_frame():
                frame_idx = st.session_state.get("frame_idx", 0)
                _cleanup_video_source()
                st.session_state["running"] = False
                video_placeholder.success(f"Completed - processed {frame_idx} frames.")
                break
            continue

        ok, frame = src.read_frame()
        if not ok or frame is None:
            frame_idx = st.session_state.get("frame_idx", 0)
            _cleanup_video_source()
            st.session_state["running"] = False
            video_placeholder.success(f"Completed - processed {frame_idx} frames.")
            break

        dt = now - prev_time
        fps = round(1 / dt, 1) if dt > 0 else 0.0
        prev_time = now

        # Only copy the frame when detection is active (raw_frame needed for clean crop).
        if show_detection:
            raw_frame = frame.copy()
        else:
            # When both are disabled, avoid the copy and just reference the current frame.
            raw_frame = frame

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
                    detection_results = pipeline.process_frame(frame)
                    draw_detections(frame, detection_results)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Detection failed on frame %d: %s", frame_idx, exc)

        saved_violations = 0
        repo = st.session_state.get("violation_repo")
        preprocessor = st.session_state.get("plate_preprocessor")
        if repo is not None and detection_results:
            for result in detection_results:
                if not result.is_violation or result.plate_text == "unknown":
                    continue
                det = result.detection
                crop = det.crop(raw_frame)
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
                    polygon=zone.polygon,
                    zone_id=zone.zone_id,
                    confidence=det.confidence,
                )
                if inserted_id is not None:
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to render frame %d: %s", frame_idx, exc)
        fps_display.metric("FPS", fps)
        frame_count_display.metric("Frame", f"{frame_idx}/{total_frames}")
        light_state_display.metric("Light State", light_state)
        timer_display.metric("Time Remaining (s)", f"{time_remaining:.1f}")
        violation_count_display.metric(
            "Violation Count",
            st.session_state.get("violation_count", 0),
        )

        elapsed = time.perf_counter() - now
        sleep_time = max(0.0, frame_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
