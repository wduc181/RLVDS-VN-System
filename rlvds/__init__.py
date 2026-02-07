"""
RLVDS-VN Main Package
=====================

Vietnam Red Light Violation Detection System

Package Structure:
    rlvds/
    ├── core/           # Core abstractions & pipeline
    ├── ingestion/      # Video input handling
    ├── detection/      # License plate detection (YOLOv5)
    ├── tracking/       # Multi-object tracking
    ├── spatial/        # Point-in-polygon, zone logic
    ├── temporal/       # Traffic light timing, violation logic
    ├── ocr/            # License plate OCR (PaddleOCR)
    ├── persistence/    # Database operations
    └── utils/          # Shared utilities
"""

__version__ = "0.1.0"
