"""
RLVDS-VN Streamlit Web Application
===================================

Mục đích:
    Entry point cho giao diện web Streamlit.
    Hiển thị video stream real-time và violation history.

Cách chạy:
    streamlit run app.py

Tham chiếu:
    - .github/sample/camera.py — logic hiển thị (cv2.imshow)
    - Ta thay cv2.imshow bằng Streamlit components

Thư viện sử dụng:
    - streamlit: Web framework
    - Các modules từ rlvds package

Layout:
    ┌─────────────────────────────────────────┐
    │  🚦 RLVDS-VN Dashboard  (Header)       │
    ├─────────────────────┬───────────────────┤
    │                     │  Traffic Light    │
    │   Video Stream      │  Status           │
    │   (st.image)        │───────────────────│
    │                     │  Zone Config      │
    ├─────────────────────┴───────────────────┤
    │  Violation History Table                │
    │  - Plate | Time | Image | Confidence    │
    └─────────────────────────────────────────┘

Pages/Components:
    1. Sidebar
       - Video source selection (file upload hoặc camera)
       - Start/Stop controls
       - Traffic light timing config
    
    2. Main Area
       - Video stream với annotations (st.image + loop)
       - Traffic light indicator (colored circle)
    
    3. Bottom
       - Violation history table (st.dataframe)
       - Statistics: total violations, unique plates, etc.

Pseudocode:
    ```
    import streamlit as st
    from config.settings import get_settings
    from rlvds.core.pipeline import Pipeline
    from rlvds.persistence.database import Database
    from rlvds.persistence.repository import ViolationRepository

    def main():
        st.set_page_config(page_title="RLVDS-VN", layout="wide")
        st.title("🚦 RLVDS-VN: Hệ thống Phát hiện Vi phạm Giao thông")

        # Sidebar controls
        with st.sidebar:
            video_file = st.file_uploader("Upload video", type=["mp4", "avi"])
            camera_id = st.number_input("Camera ID", min_value=0, value=0)
            start_btn = st.button("▶ Start")
            stop_btn = st.button("⏹ Stop")

        # Main area
        col1, col2 = st.columns([3, 1])
        with col1:
            video_placeholder = st.empty()  # cho video stream
        with col2:
            light_placeholder = st.empty()  # cho traffic light status

        # Violation history
        st.subheader("📋 Lịch sử vi phạm")
        db = Database()
        repo = ViolationRepository(db)
        violations = repo.get_all()
        st.dataframe(violations)

    if __name__ == "__main__":
        main()
    ```

TODO:
    [ ] Setup Streamlit page config
    [ ] Implement sidebar controls
    [ ] Display video stream (st.image trong loop)
    [ ] Show traffic light status
    [ ] Show violation history từ database (st.dataframe)
    [ ] Add zone drawing tool (st.canvas hoặc manual input)
    [ ] Add export violations feature (CSV download)
    [ ] Add statistics dashboard
"""

# import streamlit as st


def main():
    """Streamlit app main function."""
    # TODO: Implement theo pseudocode ở trên
    # st.set_page_config(page_title="RLVDS-VN", layout="wide")
    # st.title("🚦 RLVDS-VN: Hệ thống Phát hiện Vi phạm Giao thông")
    pass


if __name__ == "__main__":
    main()
