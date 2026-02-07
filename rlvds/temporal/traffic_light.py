"""
Traffic Light State Machine
===========================

Mục đích:
    Quản lý trạng thái đèn giao thông (giả lập hoặc từ sensor).

Thư viện sử dụng:
    - time: System timer
    - enum: State definitions

Input:
    - Cấu hình timing từ TemporalConfig
    - (Optional) Signal từ external sensor

Output:
    - Current light state: RED / YELLOW / GREEN

Classes cần implement:
    1. LightState (Enum)
       - RED
       - YELLOW
       - GREEN
    
    2. TrafficLightFSM (Finite State Machine)
       - __init__(red_sec: int, yellow_sec: int, green_sec: int)
       - start() -> None
       - get_state() -> LightState
       - get_time_remaining() -> float
       - reset() -> None
       - set_state(state: LightState) -> None  # Manual override

State Transitions:
    RED -> GREEN -> YELLOW -> RED -> ...

Timing Logic:
    - Track start_time của mỗi state
    - current_time - start_time >= duration -> transition

TODO:
    [ ] Implement LightState enum
    [ ] Implement TrafficLightFSM với time.time()
    [ ] Add callback khi state thay đổi
    [ ] (Advanced) Support external sensor input
"""
