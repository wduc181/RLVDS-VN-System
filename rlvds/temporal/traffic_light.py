"""
Traffic Light State Machine
============================

Mục đích:
    Quản lý trạng thái đèn giao thông (giả lập).
    Xác định thời điểm nào đang là đèn đỏ để phối hợp với detection.

Tham chiếu sample code:
    - .github/sample/camera.py (dòng 33-36, 53) — cycle timer logic

Thư viện sử dụng:
    - time: System timer
    - enum: State definitions

State Transitions (vòng lặp):
    RED → GREEN → YELLOW → RED → ...
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from rlvds.core.base import BaseTemporalLogic, Track
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class LightState(Enum):
    """Các trạng thái đèn giao thông."""

    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"


# Thứ tự trạng thái trong 1 cycle: RED → GREEN → YELLOW
_STATE_ORDER = (LightState.RED, LightState.GREEN, LightState.YELLOW)


class TrafficLightFSM(BaseTemporalLogic):
    """Finite State Machine giả lập chu kỳ đèn giao thông.

    Cycle layout (thời gian tuyến tính):
        ``[--- RED ---][--- GREEN ---][- YELLOW -]``

    Sử dụng modulo trên elapsed time để xác định trạng thái hiện tại.
    Tham số lấy từ ``config.settings.temporal``.

    Args:
        red_sec: Thời lượng đèn đỏ (giây).
        yellow_sec: Thời lượng đèn vàng (giây).
        green_sec: Thời lượng đèn xanh (giây).
        initial_state: Trạng thái ban đầu khi ``start()``.
    """

    def __init__(
        self,
        red_sec: int = 30,
        yellow_sec: int = 3,
        green_sec: int = 30,
        initial_state: str = "RED",
    ) -> None:
        self._red_sec = red_sec
        self._green_sec = green_sec
        self._yellow_sec = yellow_sec
        self._cycle_duration = red_sec + green_sec + yellow_sec

        # Boundaries within one cycle
        self._red_end = red_sec
        self._green_end = red_sec + green_sec

        self._initial_state = LightState(initial_state)
        self._start_time: Optional[float] = None

        # Duration lookup for convenience
        self._durations = {
            LightState.RED: red_sec,
            LightState.GREEN: green_sec,
            LightState.YELLOW: yellow_sec,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bắt đầu chu kỳ đèn. Đặt offset theo ``initial_state``."""
        now = time.time()
        # Adjust start_time so that get_state() immediately returns initial_state
        self._start_time = now - self._offset_for_state(self._initial_state)
        logger.info(
            "TrafficLightFSM started — cycle=%ds (R=%d G=%d Y=%d), initial=%s",
            self._cycle_duration,
            self._red_sec,
            self._green_sec,
            self._yellow_sec,
            self._initial_state.value,
        )

    def get_state(self) -> LightState:
        """Trả về trạng thái đèn hiện tại dựa trên elapsed time.

        Returns:
            ``LightState`` tương ứng với vị trí trong cycle.

        Raises:
            RuntimeError: Nếu FSM chưa ``start()``.
        """
        position = self._get_cycle_position()
        if position < self._red_end:
            return LightState.RED
        if position < self._green_end:
            return LightState.GREEN
        return LightState.YELLOW

    def get_time_remaining(self) -> float:
        """Thời gian còn lại (giây) của trạng thái hiện tại.

        Returns:
            Số giây còn lại trước khi chuyển sang trạng thái tiếp theo.
        """
        position = self._get_cycle_position()
        if position < self._red_end:
            return self._red_end - position
        if position < self._green_end:
            return self._green_end - position
        return self._cycle_duration - position

    def is_red(self) -> bool:
        """Shortcut kiểm tra đèn đỏ."""
        return self.get_state() == LightState.RED

    def reset(self) -> None:
        """Reset FSM về trạng thái ban đầu, bắt đầu cycle mới."""
        self.start()
        logger.debug("TrafficLightFSM reset")

    def set_state(self, state: LightState) -> None:
        """Manual override — đặt trạng thái đèn ngay lập tức.

        Điều chỉnh ``start_time`` sao cho ``get_state()`` trả về *state*
        ở đầu phase đó.

        Args:
            state: Trạng thái đèn mong muốn.
        """
        self._start_time = time.time() - self._offset_for_state(state)
        logger.info("TrafficLightFSM manual override → %s", state.value)

    # ------------------------------------------------------------------
    # BaseTemporalLogic interface
    # ------------------------------------------------------------------

    def get_light_state(self) -> str:
        """Implement ABC — trả về string state."""
        return self.get_state().value

    def update(self, elapsed: float) -> None:
        """Implement ABC — FSM dựa trên wall-clock, không cần update thủ công."""

    def is_violation(self, track: Track) -> bool:
        """Implement ABC — kiểm tra điều kiện temporal (đèn đỏ).

        Chỉ kiểm tra thành phần thời gian. Logic đầy đủ (spatial + tracking)
        nằm trong ``ViolationDetector``.

        Args:
            track: Đối tượng Track cần kiểm tra.

        Returns:
            ``True`` nếu đèn đang đỏ (điều kiện temporal thỏa).
        """
        return self.is_red()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cycle_duration(self) -> int:
        """Tổng thời gian 1 chu kỳ đèn (giây)."""
        return self._cycle_duration

    @property
    def is_started(self) -> bool:
        """FSM đã start chưa."""
        return self._start_time is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cycle_position(self) -> float:
        """Vị trí hiện tại trong cycle (0 → cycle_duration).

        Raises:
            RuntimeError: Nếu FSM chưa start.
        """
        if self._start_time is None:
            raise RuntimeError(
                "TrafficLightFSM chưa start(). Gọi start() trước khi get_state()."
            )
        elapsed = time.time() - self._start_time
        return elapsed % self._cycle_duration

    def _offset_for_state(self, state: LightState) -> float:
        """Offset (giây) từ đầu cycle tới đầu *state* phase."""
        if state == LightState.RED:
            return 0.0
        if state == LightState.GREEN:
            return float(self._red_sec)
        # YELLOW
        return float(self._red_sec + self._green_sec)
