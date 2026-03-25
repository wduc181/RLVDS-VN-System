"""
RLVDS-VN Streamlit Web Application
===================================

Má»¥c Ä‘Ã­ch:
    Entry point cho giao diá»‡n web Streamlit.
    Hiá»ƒn thá»‹ video stream real-time vÃ  violation history.

CÃ¡ch cháº¡y:
    streamlit run app.py
"""

import time
import warnings
from pathlib import Path

import cv2
import streamlit as st

# Suppress torch deprecation warnings from YOLOv5
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

from config.settings import get_settings
from rlvds.ingestion import FrameBuffer, VideoSource
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
from rlvds.core.mini_pipeline import MiniPipeline
from rlvds.detection import LicensePlateDetector
from rlvds.ocr.preprocessor import PlatePreprocessor
from rlvds.ocr.recognizer import LicensePlateOCR
from rlvds.persistence import Database, ViolationRepository

logger = get_logger(__name__)


def _list_sample_videos(samples_dir: str) -> list[str]:
    """Tráº£ vá» danh sÃ¡ch file video trong thÆ° má»¥c samples."""
    p = Path(samples_dir)
    if not p.is_dir():
        return []
    exts = {".mp4", ".avi", ".mkv", ".mov"}
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts)


def _cleanup_video_source() -> None:
    """Giải phóng VideoSource đang lưu trong session_state."""
    db = st.session_state.pop("violation_db", None)
    if db is not None:
        try:
            db.disconnect()
        except Exception as exc:
            logger.warning("Error while disconnecting database: %s", exc)

    src = st.session_state.pop("video_src", None)
    if src is not None:
        src.release()
        logger.info(
            "Video source released after %d frames",
            st.session_state.get("frame_idx", 0),
        )
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
    st.session_state.pop("violation_repo", None)
    st.session_state.pop("plate_preprocessor", None)

def _build_runtime_components() -> tuple[ViolationZone, TrafficLightFSM, ViolationDetector, FrameBuffer]:
    """Khá»Ÿi táº¡o spatial/temporal components cho phiÃªn stream hiá»‡n táº¡i."""
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
    """Streamlit app main function."""
    st.set_page_config(page_title="RLVDS-VN", layout="wide")
    st.title("ðŸš¦ RLVDS-VN â€” Video Stream Test")

    settings = get_settings()

    # â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with st.sidebar:
        st.header("Video Source")

        sample_videos = _list_sample_videos(settings.paths.samples_dir)

        if not sample_videos:
            st.warning("KhÃ´ng tÃ¬m tháº¥y video máº«u trong data/samples/")
            source_path = None
        else:
            source_path = st.selectbox(
                "Chá»n video máº«u",
                options=sample_videos,
                index=0,
                help="Chá»n file trong data/samples/",
            )

        display_width = st.slider(
            "Display width (px)", 480, 1920, 1280, step=80,
        )

        show_fps = st.checkbox("Hiá»ƒn thá»‹ FPS", value=True)
        show_zone_overlay = st.checkbox("Hiá»ƒn thá»‹ zone overlay", value=True)
        show_detection = st.checkbox(
            "Nháº­n diá»‡n biá»ƒn sá»‘",
            value=False,
            help="Báº­t detection + OCR overlay lÃªn video",
        )

        # Warning only after pipeline init attempted (mini_pipeline key exists)
        if (
            show_detection
            and st.session_state.get("running", False)
            and "mini_pipeline" in st.session_state
            and not st.session_state.get("detection_available", False)
        ):
            st.warning("Model detection chÆ°a Ä‘Æ°á»£c load. Kiá»ƒm tra detection.model_path trong config.")

        target_fps = st.slider(
            "Target FPS", 1, 60, 30,
            help="Giá»›i háº¡n tá»‘c Ä‘á»™ hiá»ƒn thá»‹ (frame/giÃ¢y)",
        )

        st.divider()
        st.subheader("Spatial Zone")
        if settings.spatial.violation_zone:
            st.caption("Vertices (x, y):")
            st.code(str(settings.spatial.violation_zone), language="python")
        else:
            st.info("`spatial.violation_zone` Ä‘ang rá»—ng. App sáº½ dÃ¹ng dummy polygon.")

        st.subheader("Traffic Light Cycle")
        st.caption(
            "R/G/Y = "
            f"{settings.temporal.red_duration_sec}/"
            f"{settings.temporal.green_duration_sec}/"
            f"{settings.temporal.yellow_duration_sec} (s)"
        )

        is_running = st.session_state.get("running", False)
        can_start = source_path is not None and not is_running

        st.button(
            "â–¶ Start",
            use_container_width=True,
            disabled=not can_start,
            on_click=lambda: st.session_state.update(should_start=True),
        )
        st.button(
            "â¹ Stop",
            use_container_width=True,
            disabled=not is_running,
            on_click=lambda: st.session_state.update(running=False),
        )

    # â”€â”€ Main area placeholders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    video_placeholder = st.empty()
    metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
    fps_display = metrics_col1.empty()
    frame_count_display = metrics_col2.empty()
    resolution_display = metrics_col3.empty()
    light_state_display = metrics_col4.empty()
    timer_display = metrics_col5.empty()
    violation_count_display = st.empty()

    # â”€â”€ Handle Start â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if st.session_state.pop("should_start", False) and source_path:
        _cleanup_video_source()

        try:
            src = VideoSource(source_path)
        except (FileNotFoundError, RuntimeError) as exc:
            st.error(f"KhÃ´ng thá»ƒ má»Ÿ video: {exc}")
            return

        w, h = src.get_frame_size()
        total_frames = src.get_frame_count()
        logger.info(
            "Streaming %s â€” %d frames, %dx%d",
            source_path, total_frames, w, h,
        )

        st.session_state["video_src"] = src
        st.session_state["frame_idx"] = 0
        st.session_state["total_frames"] = total_frames
        st.session_state["resolution"] = f"{w}Ã—{h}"
        zone, traffic_light, violation_detector, frame_buffer = _build_runtime_components()
        st.session_state["zone"] = zone
        st.session_state["traffic_light"] = traffic_light
        st.session_state["violation_detector"] = violation_detector
        st.session_state["frame_buffer"] = frame_buffer
        st.session_state["violation_count"] = 0

        # Initialize persistence for unique violation logging
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
        except Exception as exc:
            logger.warning("Failed to initialize persistence: %s", exc)
            st.session_state["violation_db"] = None
            st.session_state["violation_repo"] = None
            st.session_state["plate_preprocessor"] = None

        # Initialize detection pipeline
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
            pipeline = MiniPipeline(
                detector=detector,
                ocr=ocr_engine,
                violation_detector=violation_detector,
                crop_expand_ratio=settings.preprocessing.expand_ratio,
            )
            st.session_state["mini_pipeline"] = pipeline
            st.session_state["detection_available"] = detector.is_available()
            if detector.is_available():
                logger.info("Detection pipeline initialized successfully")
            else:
                logger.warning("Detection model not available - detection disabled")
        except Exception as exc:
            logger.warning("Failed to initialize detection pipeline: %s", exc)
            st.session_state["mini_pipeline"] = None
            st.session_state["detection_available"] = False

        st.session_state["running"] = True

    # â”€â”€ Handle Stop / cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not st.session_state.get("running", False):
        if "video_src" in st.session_state:
            _cleanup_video_source()
        video_placeholder.info("Nháº¥n **â–¶ Start** Ä‘á»ƒ báº¯t Ä‘áº§u stream video.")
        return

    # â”€â”€ Video streaming (while loop â€” no st.rerun) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    src = st.session_state.get("video_src")
    zone = st.session_state.get("zone")
    traffic_light = st.session_state.get("traffic_light")
    if src is None or not src.is_opened():
        _cleanup_video_source()
        st.session_state["running"] = False
        video_placeholder.warning("Video source khÃ´ng kháº£ dá»¥ng.")
        return
    if zone is None or traffic_light is None:
        _cleanup_video_source()
        st.session_state["running"] = False
        video_placeholder.warning("Spatial/Temporal components chÆ°a Ä‘Æ°á»£c khá»Ÿi táº¡o.")
        return

    total_frames = st.session_state.get("total_frames", 0)
    resolution_display.metric(
        "Resolution", st.session_state.get("resolution", "â€“"),
    )

    frame_interval = 1.0 / target_fps
    prev_time = time.perf_counter()

    while st.session_state.get("running", False):
        ok, frame = src.read_frame()
        if not ok or frame is None:
            frame_idx = st.session_state.get("frame_idx", 0)
            _cleanup_video_source()
            st.session_state["running"] = False
            video_placeholder.success(
                f"HoÃ n táº¥t â€” Ä‘Ã£ xá»­ lÃ½ {frame_idx} frames.",
            )
            break
        raw_frame = frame.copy()

        # FPS tÃ­nh toÃ¡n
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

        # Detection + OCR overlay
        detection_results = []
        if show_detection:
            pipeline = st.session_state.get("mini_pipeline")
            if pipeline and st.session_state.get("detection_available", False):
                try:
                    detection_results = pipeline.process_frame(frame)
                    draw_detections(frame, detection_results)
                except Exception as exc:
                    logger.error("Detection failed on frame %d: %s", frame_idx, exc)

        # Persist unique violations to SQLite + save scene/plate images
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

        # Váº½ FPS lÃªn frame náº¿u Ä‘Æ°á»£c báº­t
        if show_fps:
            draw_fps(frame, fps)

        # Resize cho hiá»ƒn thá»‹
        display_frame = set_hd_resolution(frame, width=display_width)

        # BGR â†’ RGB cho Streamlit
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # Cáº­p nháº­t UI (in-place, khÃ´ng rerun toÃ n trang)
        # Wrap in try/except to handle transient Streamlit cache errors
        try:
            video_placeholder.image(display_frame, channels="RGB")
        except Exception:
            pass  # Transient MediaFileStorageError - safe to ignore
        fps_display.metric("FPS", fps)
        frame_count_display.metric("Frame", f"{frame_idx}/{total_frames}")
        light_state_display.metric("Light State", light_state)
        timer_display.metric("Time Remaining (s)", f"{time_remaining:.1f}")
        violation_count_display.metric(
            "Violation Count (preview)",
            st.session_state.get("violation_count", 0),
        )

        # Throttle theo target FPS
        elapsed = time.perf_counter() - now
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()

