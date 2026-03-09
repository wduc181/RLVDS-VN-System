"""
RLVDS-VN Streamlit Web Application
===================================

Mục đích:
    Entry point cho giao diện web Streamlit.
    Hiển thị video stream real-time và violation history.

Cách chạy:
    streamlit run app.py
"""

import time
from pathlib import Path

import cv2
import streamlit as st

from config.settings import get_settings
from rlvds.ingestion.video_source import VideoSource
from rlvds.utils.logger import get_logger
from rlvds.utils.visualization import draw_fps, set_hd_resolution

logger = get_logger(__name__)


def _list_sample_videos(samples_dir: str) -> list[str]:
    """Trả về danh sách file video trong thư mục samples."""
    p = Path(samples_dir)
    if not p.is_dir():
        return []
    exts = {".mp4", ".avi", ".mkv", ".mov"}
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts)


def main() -> None:
    """Streamlit app main function."""
    st.set_page_config(page_title="RLVDS-VN", layout="wide")
    st.title("🚦 RLVDS-VN — Video Stream Test")

    settings = get_settings()

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Video Source")

        sample_videos = _list_sample_videos(settings.paths.samples_dir)
        source_path = st.selectbox(
            "Chọn video mẫu",
            options=sample_videos,
            index=0 if sample_videos else None,
            help="Chọn file trong data/samples/",
        )

        display_width = st.slider(
            "Display width (px)", 480, 1920, 1280, step=80,
        )

        show_fps = st.checkbox("Hiển thị FPS", value=True)

        start = st.button("▶ Start", use_container_width=True)
        stop = st.button("⏹ Stop", use_container_width=True)

    # ── Main area ────────────────────────────────────────────────────
    video_placeholder = st.empty()
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    fps_display = metrics_col1.empty()
    frame_count_display = metrics_col2.empty()
    resolution_display = metrics_col3.empty()

    # ── State management ─────────────────────────────────────────────
    if stop:
        st.session_state["running"] = False

    if start and source_path:
        st.session_state["running"] = True

    if not st.session_state.get("running", False):
        st.info("Nhấn **▶ Start** để bắt đầu stream video.")
        return

    # ── Video loop ───────────────────────────────────────────────────
    if not source_path:
        st.warning("Không tìm thấy video mẫu trong data/samples/")
        return

    try:
        src = VideoSource(source_path)
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(f"Không thể mở video: {exc}")
        return

    w, h = src.get_frame_size()
    resolution_display.metric("Resolution", f"{w}×{h}")
    total_frames = src.get_frame_count()
    logger.info("Streaming %s — %d frames, %dx%d", source_path, total_frames, w, h)

    prev_time = time.perf_counter()
    frame_idx = 0

    try:
        for frame in src:
            if not st.session_state.get("running", False):
                break

            # FPS tính toán
            now = time.perf_counter()
            dt = now - prev_time
            fps = int(1 / dt) if dt > 0 else 0
            prev_time = now
            frame_idx += 1

            # Vẽ FPS lên frame nếu được bật
            if show_fps:
                draw_fps(frame, fps)

            # Resize cho hiển thị
            display_frame = set_hd_resolution(frame, width=display_width)

            # BGR → RGB cho Streamlit
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Cập nhật UI
            video_placeholder.image(display_frame, channels="RGB")
            fps_display.metric("FPS", fps)
            frame_count_display.metric("Frame", f"{frame_idx}/{total_frames}")
    finally:
        src.release()
        logger.info("Video source released after %d frames", frame_idx)

    st.session_state["running"] = False
    st.success(f"Hoàn tất — đã xử lý {frame_idx} frames.")


if __name__ == "__main__":
    main()
