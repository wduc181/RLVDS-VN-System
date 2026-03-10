"""
Tests for Spatial Module
========================

Covers:
    - rlvds/spatial/polygon.py  (create_polygon, create_mask, point_in_polygon, ...)
    - rlvds/spatial/zones.py    (ViolationZone)

Chạy: pytest tests/test_polygon.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from rlvds.spatial.polygon import (
    create_mask,
    create_polygon,
    draw_polygon,
    point_distance_to_polygon,
    point_in_polygon,
)
from rlvds.spatial.zones import ViolationZone


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def square_vertices() -> list[list[int]]:
    """Hình vuông 100×100 tại gốc."""
    return [[0, 0], [100, 0], [100, 100], [0, 100]]


@pytest.fixture
def square_polygon(square_vertices: list[list[int]]) -> np.ndarray:
    """Polygon array từ hình vuông 100×100."""
    return create_polygon(square_vertices)


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Frame trắng 200×200×3."""
    return np.ones((200, 200, 3), dtype=np.uint8) * 255


# =========================================================================
# test create_polygon
# =========================================================================

class TestCreatePolygon:
    """Test ``create_polygon()``."""

    def test_shape(self, square_vertices: list[list[int]]) -> None:
        poly = create_polygon(square_vertices)
        assert poly.shape == (4, 1, 2)
        assert poly.dtype == np.int32

    def test_triangle(self) -> None:
        tri = create_polygon([[0, 0], [10, 0], [5, 10]])
        assert tri.shape == (3, 1, 2)

    def test_too_few_vertices_raises(self) -> None:
        with pytest.raises(ValueError, match="ít nhất 3 đỉnh"):
            create_polygon([[0, 0], [10, 0]])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            create_polygon([])


# =========================================================================
# test point_in_polygon
# =========================================================================

class TestPointInPolygon:
    """Test ``point_in_polygon()``."""

    def test_inside(self, square_polygon: np.ndarray) -> None:
        assert point_in_polygon((50.0, 50.0), square_polygon) is True

    def test_outside(self, square_polygon: np.ndarray) -> None:
        assert point_in_polygon((150.0, 150.0), square_polygon) is False

    def test_on_edge(self, square_polygon: np.ndarray) -> None:
        # Trên cạnh trái (x=0)
        assert point_in_polygon((0.0, 50.0), square_polygon) is True

    def test_on_vertex(self, square_polygon: np.ndarray) -> None:
        assert point_in_polygon((0.0, 0.0), square_polygon) is True

    def test_just_outside(self, square_polygon: np.ndarray) -> None:
        assert point_in_polygon((101.0, 50.0), square_polygon) is False


# =========================================================================
# test point_distance_to_polygon
# =========================================================================

class TestPointDistanceToPolygon:
    """Test ``point_distance_to_polygon()``."""

    def test_inside_positive(self, square_polygon: np.ndarray) -> None:
        dist = point_distance_to_polygon((50.0, 50.0), square_polygon)
        assert dist > 0

    def test_outside_negative(self, square_polygon: np.ndarray) -> None:
        dist = point_distance_to_polygon((150.0, 150.0), square_polygon)
        assert dist < 0

    def test_on_edge_zero(self, square_polygon: np.ndarray) -> None:
        dist = point_distance_to_polygon((0.0, 50.0), square_polygon)
        assert dist == pytest.approx(0.0, abs=1e-6)


# =========================================================================
# test create_mask
# =========================================================================

class TestCreateMask:
    """Test ``create_mask()``."""

    def test_outside_zone_is_black(self, blank_frame: np.ndarray) -> None:
        poly = create_polygon([[50, 50], [150, 50], [150, 150], [50, 150]])
        masked = create_mask(blank_frame, poly)
        # Pixel ngoài zone (0,0) phải là đen
        assert np.all(masked[0, 0] == 0)

    def test_inside_zone_preserved(self, blank_frame: np.ndarray) -> None:
        poly = create_polygon([[50, 50], [150, 50], [150, 150], [50, 150]])
        masked = create_mask(blank_frame, poly)
        # Pixel trong zone (100,100) phải giữ nguyên (trắng)
        assert np.all(masked[100, 100] == 255)

    def test_output_shape_matches_input(self, blank_frame: np.ndarray) -> None:
        poly = create_polygon([[10, 10], [50, 10], [50, 50], [10, 50]])
        masked = create_mask(blank_frame, poly)
        assert masked.shape == blank_frame.shape


# =========================================================================
# test draw_polygon
# =========================================================================

class TestDrawPolygon:
    """Test ``draw_polygon()``."""

    def test_modifies_frame(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        poly = create_polygon([[10, 10], [90, 10], [90, 90], [10, 90]])
        result = draw_polygon(frame, poly, color=(0, 255, 0))
        # Frame phải có pixel khác 0 (viền đã vẽ)
        assert np.any(result != 0)

    def test_returns_same_reference(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        poly = create_polygon([[10, 10], [90, 10], [90, 90], [10, 90]])
        result = draw_polygon(frame, poly)
        # draw_polygon thay đổi in-place, trả về cùng reference
        assert result is frame


# =========================================================================
# test ViolationZone
# =========================================================================

class TestViolationZone:
    """Test ``ViolationZone``."""

    def test_contains_inside(self, square_vertices: list[list[int]]) -> None:
        zone = ViolationZone(vertices=square_vertices, zone_id="test")
        assert zone.contains((50.0, 50.0)) is True

    def test_contains_outside(self, square_vertices: list[list[int]]) -> None:
        zone = ViolationZone(vertices=square_vertices, zone_id="test")
        assert zone.contains((150.0, 150.0)) is False

    def test_is_in_zone_alias(self, square_vertices: list[list[int]]) -> None:
        zone = ViolationZone(vertices=square_vertices, zone_id="test")
        assert zone.is_in_zone((50.0, 50.0)) == zone.contains((50.0, 50.0))

    def test_set_zone_updates_polygon(self) -> None:
        zone = ViolationZone(vertices=[[0, 0], [10, 0], [10, 10], [0, 10]])
        # Ban đầu (50,50) nằm ngoài zone nhỏ
        assert zone.contains((50.0, 50.0)) is False
        # Mở rộng zone
        zone.set_zone([(0, 0), (100, 0), (100, 100), (0, 100)])
        assert zone.contains((50.0, 50.0)) is True

    def test_apply_mask(
        self,
        square_vertices: list[list[int]],
        blank_frame: np.ndarray,
    ) -> None:
        zone = ViolationZone(vertices=square_vertices)
        masked = zone.apply_mask(blank_frame)
        assert masked.shape == blank_frame.shape
        # Pixel ngoài zone phải là đen
        assert np.all(masked[150, 150] == 0)

    def test_draw(self, square_vertices: list[list[int]]) -> None:
        zone = ViolationZone(vertices=square_vertices)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = zone.draw(frame)
        assert np.any(result != 0)

    def test_properties(self, square_vertices: list[list[int]]) -> None:
        zone = ViolationZone(
            vertices=square_vertices, zone_id="cam01",
        )
        assert zone.zone_id == "cam01"
        assert zone.vertices == square_vertices
        assert zone.polygon.shape == (4, 1, 2)
