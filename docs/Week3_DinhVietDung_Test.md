# Week 3 - Test Guide (Task Ðinh Vi?t Dung)

## Scope
- OCR pipeline d?c l?p
- Mock violation check: `RED + inside polygon => violation`
- Lu?ng mini pipeline: `Video/Frame -> Detect -> OCR -> Violation(mock)`

## 1) Cài test dependency
```bash
pip install pytest
```

## 2) Ch?y test module Ðinh Vi?t Dung (mock)
```bash
python -m pytest -q tests/test_ocr_pipeline.py
```

Test file này dùng fake detector/fake OCR engine nên không ph? thu?c model weights.

## 3) Ch?y test n?n dã có (spatial + temporal)
```bash
python -m pytest -q tests/test_polygon.py tests/test_traffic_light.py
```

## 4) Smoke test nhanh OCR parser (không c?n PaddleOCR th?t)
```bash
python - <<'PY'
import numpy as np
from rlvds.ocr.recognizer import LicensePlateOCR

class FakeOCR:
    def ocr(self, _img):
        return [[[[0,0],[1,0],[1,1],[0,1]], ('30A12345', 0.95)]]

ocr = LicensePlateOCR(ocr_engine=FakeOCR())
print(ocr.recognize(np.ones((20,60,3), dtype=np.uint8) * 255))
PY
```
Expected: `30A-12345`

## 5) Test mini pipeline v?i detector th?t (sau khi module detector hoàn thi?n)
```python
from rlvds.core.mini_pipeline import MiniPipeline
from rlvds.detection.detector import LicensePlateDetector
from rlvds.ocr.recognizer import LicensePlateOCR
from rlvds.spatial.zones import ViolationZone
from rlvds.temporal.traffic_light import TrafficLightFSM
from rlvds.temporal.violation import ViolationDetector

# Init components
detector = LicensePlateDetector(model_path="weights/lp_vn_det_yolov5n.pt", device="cpu")
ocr = LicensePlateOCR(lang="en", use_gpu=False)
zone = ViolationZone(vertices=[[0,0],[1000,0],[1000,700],[0,700]])
fsm = TrafficLightFSM(initial_state="RED")
fsm.start()
vd = ViolationDetector(zone=zone, traffic_light=fsm)

pipeline = MiniPipeline(detector=detector, ocr=ocr, violation_detector=vd)
results = pipeline.run_video("data/samples/sample.mp4", max_frames=200)
print("results:", len(results))
print("violations:", sum(int(r.is_violation) for r in results))
```

## 6) Criteria pass cho task tu?n 3 (ph?n Ðinh Vi?t Dung)
- OCR module ch?y d?c l?p và tr? `unknown` khi confidence th?p/không d?c du?c.
- Mock violation check ho?t d?ng dúng theo rule `RED + in-zone`.
- Có th? ghép frame flow `Detect -> OCR -> Violation(mock)` qua `MiniPipeline`.
