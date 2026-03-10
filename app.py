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


def _cleanup_video_source() -> None:
    """Giải phóng VideoSource đang lưu trong session_state."""
    src = st.session_state.pop("video_src", None)
    if src is not None:
        src.release()
        logger.info(
            "Video source released after %d frames",
            st.session_state.get("frame_idx", 0),
        )
    st.session_state.pop("frame_idx", None)
    st.session_state.pop("prev_time", None)
    st.session_state.pop("total_frames", None)
    st.session_state.pop("resolution", None)


def main() -> None:
    """Streamlit app main function."""
    st.set_page_config(page_title="RLVDS-VN", layout="wide")
    st.title("🚦 RLVDS-VN — Video Stream Test")

    settings = get_settings()

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Video Source")

        sample_videos = _list_sample_videos(settings.paths.samples_dir)

        if not sample_videos:
            st.warning("Không tìm thấy video mẫu trong data/samples/")
            source_path = None
        else:
            source_path = st.selectbox(
                "Chọn video mẫu",
                options=sample_videos,
                index=0,
                help="Chọn file trong data/samples/",
            )

        display_width = st.slider(
            "Display width (px)", 480, 1920, 1280, step=80,
        )

        show_fps = st.checkbox("Hiển thị FPS", value=True)

        target_fps = st.slider(
            "Target FPS", 1, 60, 30,
            help="Giới hạn tốc độ hiển thị (frame/giây)",
        )

        can_start = source_path is not None and not st.session_state.get(
            "running", False
        )
        start = st.button(
            "▶ Start", use_container_width=True, disabled=not can_start,
        )
        stop = st.button(
            "⏹ Stop",
            use_container_width=True,
            disabled=not st.session_state.get("running", False),
        )

    # ── Main area ────────────────────────────────────────────────────
    video_placeholder = st.empty()
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    fps_display = metrics_col1.empty()
    frame_count_display = metrics_col2.empty()
    resolution_display = metrics_col3.empty()

    # ── Handle Stop ──────────────────────────────────────────────────
    if stop:
        _cleanup_video_source()
        st.session_state["running"] = False
        st.rerun()

    # ── Handle Start ─────────────────────────────────────────────────
    if start and source_path:
        # Dọn dẹp source cũ nếu có
        _cleanup_video_source()

        try:
            src = VideoSource(source_path)
        except (FileNotFoundError, RuntimeError) as exc:
            st.error(f"Không thể mở video: {exc}")
            return

        w, h = src.get_frame_size()
        total_frames = src.get_frame_count()
        logger.info(
            "Streaming %s — %d frames, %dx%d",
            source_path, total_frames, w, h,
        )

        # Persist vào session_state
        st.session_state["video_src"] = src
        st.session_state["frame_idx"] = 0
        st.session_state["prev_time"] = time.perf_counter()
        st.session_state["total_frames"] = total_frames
        st.session_state["resolution"] = f"{w}×{h}"
        st.session_state["running"] = True
        st.rerun()

    # ── Idle state ───────────────────────────────────────────────────
    if not st.session_state.get("running", False):
        st.info("Nhấn **▶ Start** để bắt đầu stream video.")
        return

    # ── Render 1 frame per rerun ─────────────────────────────────────
    src: VideoSource = st.session_state.get("video_src")  # type: ignore[assignment]
    if src is None or not src.is_opened():
        _cleanup_video_source()
        st.session_state["running"] = False
        st.warning("Video source không khả dụng.")
        return

    resolution_display.metric(
        "Resolution", st.session_state.get("resolution", "–"),
    )
    total_frames: int = st.session_state.get("total_frames", 0)  # type: ignore[no-redef]

    ok, frame = src.read_frame()

    if not ok or frame is None:
        # Hết video
        _cleanup_video_source()
        st.session_state["running"] = False
        frame_idx = st.session_state.get("frame_idx", 0)
        st.success(f"Hoàn tất — đã xử lý {frame_idx} frames.")
        return

    # FPS tính toán
    now = time.perf_counter()
    prev_time = st.session_state.get("prev_time", now)
    dt = now - prev_time
    fps = int(1 / dt) if dt > 0 else 0
    st.session_state["prev_time"] = now

    frame_idx = st.session_state.get("frame_idx", 0) + 1
    st.session_state["frame_idx"] = frame_idx

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

    # Throttle theo target FPS để tránh busy-loop
    frame_interval = 1.0 / target_fps
    elapsed = time.perf_counter() - now
    sleep_time = frame_interval - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

    # Trigger next rerun để đọc frame tiếp theo
    st.rerun()


if __name__ == "__main__":
    main()
