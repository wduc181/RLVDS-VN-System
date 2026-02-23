"""
RLVDS-VN Main Entry Point (CLI/Pipeline)
=========================================

Mục đích:
    Entry point cho chạy pipeline từ command line.
    Parse arguments, load config, khởi tạo Pipeline, và chạy.

Cách chạy:
    python main.py --video data/samples/test.mp4
    python main.py --camera 0
    python main.py --video data/samples/test.mp4 --debug

Tham chiếu:
    - .github/sample/camera.py — script gốc chạy trực tiếp
    - rlvds/core/pipeline.py — Pipeline class đã modular hóa

Thư viện sử dụng:
    - argparse: CLI arguments
    - config.settings: get_settings()
    - rlvds.core.pipeline: Pipeline

Arguments:
    --video: Path to video file (ví dụ: data/samples/test.mp4)
    --camera: Camera ID (0 for webcam, default)
    --config: Path to config file (default: config/default.yaml)
    --debug: Enable debug mode (hiển thị thêm thông tin)

Flow:
    1. Parse arguments (argparse)
    2. Load config: settings = get_settings()
    3. Override config nếu có args (video source, debug...)
    4. Khởi tạo Pipeline: pipeline = Pipeline(settings)
    5. Chạy pipeline:
       - Nếu --video: pipeline.run(args.video)
       - Nếu --camera: pipeline.run(int(args.camera))
    6. Handle Ctrl+C: pipeline.stop()

Pseudocode:
    ```
    import argparse
    from config.settings import get_settings
    from rlvds.core.pipeline import Pipeline

    def main():
        parser = argparse.ArgumentParser(description="RLVDS-VN: Red Light Violation Detection")
        parser.add_argument("--video", type=str, help="Path to video file")
        parser.add_argument("--camera", type=int, default=None, help="Camera ID")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        args = parser.parse_args()

        settings = get_settings()
        if args.debug:
            settings.debug = True

        pipeline = Pipeline(settings)

        source = args.video if args.video else (args.camera if args.camera is not None else 0)
        
        try:
            pipeline.run(source)
        except KeyboardInterrupt:
            print("\\nStopping pipeline...")
        finally:
            pipeline.stop()

    if __name__ == "__main__":
        main()
    ```

TODO:
    [ ] Import argparse, get_settings, Pipeline
    [ ] Implement argument parsing
    [ ] Load config và override theo args
    [ ] Khởi tạo Pipeline
    [ ] Handle Ctrl+C gracefully
    [ ] Add tqdm progress bar (optional)
"""


def main():
    """Main entry point."""
    # TODO: Implement theo pseudocode ở trên
    pass


if __name__ == "__main__":
    main()
