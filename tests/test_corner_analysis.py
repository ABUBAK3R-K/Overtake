"""Tests for the corner/mini-sector driver-performance module (PRD FR-9)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.storage import table_exists
from src.performance.corner import (
    CORNER_TELEMETRY_COLUMNS,
    _corner_windows,
    analyze_corner_performance,
)

BAHRAIN_INGESTED = table_exists("races", "2023_bahrain")


# ─── Pure-function unit tests (no FastF1 session needed) ──────────────────────

class TestCornerWindows:
    def test_windows_cover_each_distance_and_stay_ordered(self):
        distances = [100.0, 400.0, 900.0, 1600.0]
        lap_length = 2000.0
        windows = _corner_windows(distances, lap_length)
        assert len(windows) == len(distances)
        for (start, end), d in zip(windows, distances):
            assert start <= d <= end

    def test_windows_are_clipped_to_lap_bounds(self):
        # Large margins relative to the small first/last gaps force the
        # unclipped window to go below 0 (corner 0) and above lap_length
        # (corner 1) — this actually exercises the clipping, unlike a
        # generous max_margin on evenly-spaced corners which never hits it.
        distances = [5.0, 1995.0]
        lap_length = 2000.0
        windows = _corner_windows(distances, lap_length, margin_frac=1.0, max_margin=1000.0)
        assert windows[0][0] == 0.0    # clipped from 5 - 10 = -5
        assert windows[1][1] == lap_length  # clipped from 1995 + 10 = 2005
        for start, end in windows:
            assert start >= 0.0
            assert end <= lap_length

    def test_margin_respects_neighbour_spacing(self):
        """Two corners close together shouldn't get overlapping-by-a-lot
        windows that swallow the whole gap between them."""
        distances = [500.0, 520.0, 1500.0]  # first two only 20m apart
        windows = _corner_windows(distances, lap_length=2000.0)
        # window for corner 0 shouldn't extend past corner 1's own position
        assert windows[0][1] <= distances[1] + 1e-6


# ─── analyze_corner_performance: synthetic data, no FastF1 needed ─────────────

def _ct(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CORNER_TELEMETRY_COLUMNS)


class TestAnalyzeCornerPerformance:
    def _row(self, corner_number, segment_time_s, min_speed=150.0, throttle_pct=80.0,
              brake_point=20.0, gear=4):
        return {
            "corner_number": corner_number, "corner_label": str(corner_number),
            "min_speed": min_speed, "throttle_pct": throttle_pct,
            "brake_point": brake_point, "gear": gear,
            "segment_time_s": segment_time_s, "window_start_m": 0.0, "window_end_m": 100.0,
        }

    def test_positive_time_lost_when_driver_slower(self):
        driver = _ct([self._row(1, segment_time_s=2.5)])
        benchmark = _ct([self._row(1, segment_time_s=2.0)])
        result = analyze_corner_performance(driver, benchmark)
        assert result["n_corners"] == 1
        assert result["corners"][0]["time_lost_s"] == pytest.approx(0.5)
        assert result["total_time_lost_s"] == pytest.approx(0.5)

    def test_negative_time_lost_when_driver_faster(self):
        driver = _ct([self._row(1, segment_time_s=1.8)])
        benchmark = _ct([self._row(1, segment_time_s=2.0)])
        result = analyze_corner_performance(driver, benchmark)
        assert result["corners"][0]["time_lost_s"] == pytest.approx(-0.2)

    def test_worst_and_best_corner_identified(self):
        driver = _ct([self._row(1, 2.5), self._row(2, 1.5)])
        benchmark = _ct([self._row(1, 2.0), self._row(2, 2.0)])
        result = analyze_corner_performance(driver, benchmark)
        assert result["worst_corner"]["corner_number"] == 1
        assert result["best_corner"]["corner_number"] == 2

    def test_empty_inputs_return_empty_result(self):
        empty = _ct([])
        result = analyze_corner_performance(empty, empty)
        assert result == {
            "corners": [], "total_time_lost_s": 0.0,
            "worst_corner": None, "best_corner": None, "n_corners": 0,
        }

    def test_missing_brake_point_becomes_none_not_nan(self):
        driver = _ct([self._row(1, 2.0, brake_point=None)])
        benchmark = _ct([self._row(1, 2.0, brake_point=15.0)])
        result = analyze_corner_performance(driver, benchmark)
        assert result["corners"][0]["brake_point_delta_m"] is None

    def test_only_matching_corner_numbers_are_compared(self):
        """A corner present in only one of the two laps is dropped by the
        inner join, not treated as a 0 (which would silently understate or
        overstate total_time_lost_s)."""
        driver = _ct([self._row(1, 2.0), self._row(2, 3.0)])
        benchmark = _ct([self._row(1, 2.0)])
        result = analyze_corner_performance(driver, benchmark)
        assert result["n_corners"] == 1
        assert result["corners"][0]["corner_number"] == 1


# ─── Integration: real cached FastF1 session ───────────────────────────────────

@pytest.mark.skipif(not BAHRAIN_INGESTED, reason="2023_bahrain not ingested")
class TestRealSession:
    def test_build_corner_telemetry_matches_circuit_corner_count(self):
        from src.performance.corner import _cached_session, build_corner_telemetry, get_corner_boundaries
        session = _cached_session("2023_bahrain")
        boundaries = get_corner_boundaries(session)
        ct = build_corner_telemetry(session, "VER", 20)
        assert not ct.empty
        assert len(ct) <= len(boundaries)  # every window should produce at most one row
        assert set(ct.columns) == set(CORNER_TELEMETRY_COLUMNS)
        assert (ct["segment_time_s"] > 0).all()
        assert (ct["min_speed"] > 0).all()

    def test_unknown_lap_raises(self):
        from src.performance.corner import _cached_session, build_corner_telemetry
        session = _cached_session("2023_bahrain")
        with pytest.raises(ValueError):
            build_corner_telemetry(session, "VER", 9999)

    def test_analyze_driver_lap_vs_benchmark_end_to_end(self):
        from src.performance.corner import analyze_driver_lap_vs_benchmark
        result = analyze_driver_lap_vs_benchmark("2023_bahrain", "VER", 20)
        assert result["race_id"] == "2023_bahrain"
        assert result["driver"] == "VER"
        assert result["lap"] == 20
        assert result["n_corners"] > 0
        assert result["benchmark_driver"] is not None
        assert result["benchmark_lap"] is not None
        for c in result["corners"]:
            assert "time_lost_s" in c

    def test_explicit_benchmark_is_used(self):
        from src.performance.corner import analyze_driver_lap_vs_benchmark
        result = analyze_driver_lap_vs_benchmark(
            "2023_bahrain", "VER", 20, benchmark_driver="HAM", benchmark_lap=25,
        )
        assert result["benchmark_driver"] == "HAM"
        assert result["benchmark_lap"] == 25


@pytest.mark.skipif(not BAHRAIN_INGESTED, reason="2023_bahrain not ingested")
def test_corner_analysis_api_endpoint():
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    resp = client.get("/api/corner-analysis/2023_bahrain/VER/20")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_corners"] > 0
    assert "corners" in data
