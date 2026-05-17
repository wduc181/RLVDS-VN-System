"""
RLVDS-VN Main Entry Point (CLI)
================================

Usage::

    python main.py --video data/samples/test.mp4
    python main.py --camera 0
    python main.py --video data/samples/test.mp4 --debug --no-display
"""

from __future__ import annotations

import argparse
import sys

from config.settings import get_settings
from rlvds.core.pipeline import Pipeline
from rlvds.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RLVDS-VN: Red Light Violation Detection System"
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--video", type=str, help="Path to video file"
    )
    source_group.add_argument(
        "--camera", type=int, default=None, help="Camera device index (e.g. 0)"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML file"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable OpenCV display window"
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.debug:
        settings.debug = True
        setup_logger(name="rlvds", level="DEBUG")

    source = args.video if args.video else (args.camera if args.camera is not None else 0)

    pipeline = Pipeline(settings)

    try:
        pipeline.run(source, display=not args.no_display)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
