"""
Violation Detection Logic
=========================

Mục đích:
    Kết hợp Spatial (zone) + Temporal (light) để xác định vi phạm.

Violation Condition:
    (Light State == RED) AND (Vehicle in Violation Zone) = VIOLATION

Input:
    - track: Track (từ tracker)
    - light_state: LightState (từ traffic light FSM)
    - zone: ViolationZone (từ spatial module)

Output:
    - is_violation: bool
    - violation_info: dict (nếu có vi phạm)

Classes cần implement:
    1. ViolationDetector(BaseTemporalLogic)
       - __init__(zone: ViolationZone, traffic_light: TrafficLightFSM)
       - check_violation(track: Track) -> bool
       - get_violation_info(track: Track) -> dict | None
       - get_light_state() -> LightState

Violation Info structure:
    {
        "track_id": int,
        "timestamp": str,
        "light_state": str,
        "zone_id": str,
        "bbox": tuple,
        "confidence": float
    }

TODO:
    [ ] Implement check_violation() logic
    [ ] Tránh duplicate violations cho cùng 1 track
    [ ] Add cooldown period giữa các violations
    [ ] Log violations cho debugging
"""
