"""
Traffic Light State Machine
============================

Mục đích:
    Quản lý trạng thái đèn giao thông (giả lập).
    Xác định thời điểm nào đang là đèn đỏ để phối hợp với detection.

Tham chiếu sample code:
    - .github/sample/camera.py (dòng 33-36, 53) — cycle timer logic

    camera.py logic:
      start_time = time.time()
      plate_detection_duration = 30    # 30s đèn đỏ → detect + ghi nhận vi phạm
      cycle_duration = 60              # 60s tổng chu kỳ

      # Trong main loop:
      elapsed_time = current_time - start_time
      if elapsed_time % cycle_duration < plate_detection_duration:
          # → Đang đèn đỏ → phát hiện vi phạm
      else:
          # → Đang đèn xanh → xử lý dữ liệu

Thư viện sử dụng:
    - time: System timer
    - enum: State definitions

===========================================================================
Classes cần implement:
===========================================================================

1. LightState (Enum)
   Các trạng thái đèn:
     - RED     = "RED"
     - YELLOW  = "YELLOW"
     - GREEN   = "GREEN"

2. TrafficLightFSM (Finite State Machine)
   - __init__(red_sec: int = 30, yellow_sec: int = 3, green_sec: int = 30,
              initial_state: str = "RED")
     + Lưu durations cho mỗi trạng thái
     + Tính cycle_duration = red_sec + yellow_sec + green_sec
     + Lưu start_time = None (chưa start)

   - start() -> None
     + Ghi nhận start_time = time.time()

   - get_state() -> LightState
     + Tính elapsed = time.time() - start_time
     + Tính vị trí trong cycle: position = elapsed % cycle_duration
     + Logic:
       if position < red_sec:
           return LightState.RED
       elif position < red_sec + green_sec:
           return LightState.GREEN
       else:
           return LightState.YELLOW

     Lưu ý: camera.py đơn giản hóa thành 2 trạng thái (red/green),
     ta mở rộng thêm yellow theo TemporalConfig

   - get_time_remaining() -> float
     + Tính thời gian còn lại của trạng thái hiện tại

   - is_red() -> bool
     + Shortcut: return self.get_state() == LightState.RED

   - reset() -> None
     + Reset start_time về time.time() hiện tại

   - set_state(state: LightState) -> None
     + Manual override — điều chỉnh start_time sao cho get_state() trả về state mong muốn

State Transitions (vòng lặp):
    RED → GREEN → YELLOW → RED → ...

Config (từ config/settings.py):
    temporal:
      red_duration_sec: 30
      green_duration_sec: 30
      yellow_duration_sec: 3
      initial_state: "RED"

TODO:
    [ ] Import time, enum
    [ ] Implement LightState enum
    [ ] Implement TrafficLightFSM
    [ ] Implement get_state() với modulo cycle logic
    [ ] Implement get_time_remaining()
    [ ] Implement is_red() shortcut
    [ ] Implement reset(), set_state()
    [ ] Test: FSM start → sleep 31s → kiểm tra state chuyển sang GREEN
"""
