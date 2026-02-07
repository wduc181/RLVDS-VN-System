# RLVDS-VN: Vietnam Red Light Violation Detection System

> **Note:** Đây là một **Learning Project** (Dự án học tập), được phát triển nhằm mục đích nghiên cứu và học tập.

## Mô tả dự án

Hệ thống thị giác máy tính tự động phát hiện hành vi **vượt đèn đỏ** và nhận diện biển số xe máy, ô tô tại Việt Nam dựa trên kết hợp giữa không gian (polygon) và thời gian (timing).

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python 3.10 |
| Detection | YOLOv5 |
| OCR | PaddleOCR (ppOCRv4) |
| Tracking | SORT/ByteTrack |
| Giao diện | Streamlit |
| Database | SQLite |
| Config | Pydantic + YAML |

## Cấu trúc dự án

```
RLVDS-VN-System/
├── app.py                      # Streamlit UI entry point
├── main.py                     # CLI/Pipeline entry point
├── config/                     # Configuration
│   ├── __init__.py
│   ├── default.yaml            # Default config values
│   └── settings.py             # Pydantic settings
│
├── rlvds/                      # Main Python package
│   ├── core/                   # Core abstractions & pipeline
│   │   ├── base.py             # Abstract base classes
│   │   └── pipeline.py         # Pipeline orchestrator
│   │
│   ├── ingestion/              # Data Ingestion Layer
│   │   ├── video_source.py     # Video/Camera input
│   │   └── frame_buffer.py     # Frame buffering
│   │
│   ├── detection/              # Object Detection Layer
│   │   ├── detector.py         # YOLOv5 detector
│   │   └── models.py           # Detection dataclasses
│   │
│   ├── tracking/               # Object Tracking Layer
│   │   ├── tracker.py          # Multi-object tracker
│   │   └── track_state.py      # Track lifecycle
│   │
│   ├── spatial/                # Spatial Reasoning Layer
│   │   ├── polygon.py          # Point-in-polygon
│   │   ├── zones.py            # Violation zones
│   │   └── calibration.py      # Camera calibration
│   │
│   ├── temporal/               # Temporal Logic Layer
│   │   ├── traffic_light.py    # Traffic light FSM
│   │   ├── timing.py           # Timing sync
│   │   └── violation.py        # Violation detection
│   │
│   ├── ocr/                    # OCR Layer
│   │   ├── recognizer.py       # PaddleOCR wrapper
│   │   └── postprocess.py      # Text cleanup
│   │
│   ├── persistence/            # Persistence Layer
│   │   ├── database.py         # SQLite operations
│   │   ├── models.py           # Data models
│   │   └── repository.py       # Data access
│   │
│   └── utils/                  # Utilities
│       ├── logger.py           # Logging
│       ├── visualization.py    # Drawing utils
│       └── io.py               # File I/O
│
├── weights/                    # Model weights (.pt files)
├── data/                       # Data storage
│   ├── samples/                # Sample videos
│   └── violations/             # Captured images
├── tests/                      # Test suite
├── docs/                       # Documentation
└── requirements.txt            # Dependencies
```

## Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                  CameraAI                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  OpenCV   ┌─────────────┐  YOLOv5 Detect  ┌─────────────┐   │
│  │   Video     │──────────▶│   Frame     │────────────────▶│   Plate     │   │
│  │  Realtime   │           │   Image     │  License Plate  │   Image     │   │
│  └─────────────┘           └──────┬──────┘                 └──────┬──────┘   │
│                                   │                               │          │
│                                   │ Save                          │ Pre-     │
│                                   ▼                               │ process  │
│  ┌─────────────┐  Update   ┌─────────────┐  Clean Data            ▼          │
│  │  Database   │◀──────────│    Data     │◀─────────────  ┌─────────────┐    │
│  │  (SQLite)   │           │(Text,Time,  │                │  ppOCRv4    │    │
│  └─────────────┘           │    Img)     │◀───────────────│  Text Plate │    │
│                            └─────────────┘   Check Valid  └─────────────┘    │
│                                   ▲               Save                       │
│                                   └───────────────────────────────────────   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Hướng dẫn cài đặt

```bash
# 1. Clone repository
git clone <repo-url>
cd RLVDS-VN-System

# 2. Tạo conda environment
conda create -n rlvds python=3.10 -y
conda activate rlvds

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Download model weights
# (Đang cập nhật)
```

## Cách sử dụng (Đang cập nhật)
