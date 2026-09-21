"""RaceState data structure representing the full state of the race at the end of a lap.

Interface contract per Design.md Section 5 and PROJECT_STATUS.md Section 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RaceState:
    """Snapshot of the race state at the completion of `lap`.

    Attributes:
        race_id: Identifier of the race (e.g. '2023_bahrain')
        lap: The lap just completed (1-indexed)
        positions: Driver code -> current position (1 to N)
        gaps: Driver code -> gap to leader in seconds (0.0 for leader)
        tyres: Driver code -> (compound, tyre_age_in_laps)
        safety_car: True if Safety Car or Virtual Safety Car was active during the lap
        total_laps: Total scheduled race laps
        circuit: Circuit name (e.g. 'Sakhir', 'Monaco')
        status: High-level race status ('RACING', 'SAFETY_CAR', 'VIRTUAL_SAFETY_CAR', 'RED_FLAG', 'FINISHED')
        weather: Current weather details {'track_temp': float, 'air_temp': float, 'is_wet': bool}
        last_lap_times: Driver code -> last completed lap time in seconds
        pit_stops_count: Driver code -> total number of pit stops completed so far
        intervals: Driver code -> gap to the car immediately ahead in seconds (0.0 for leader)
        retired: Driver code -> retirement description/reason
        fastest_lap: Overall fastest lap set up to this lap {'driver': str, 'lap': int, 'lap_time': float}
        pit_stops_history: Chronological list of pit stops completed up to this lap
        race_control_events: Chronological list of race control events up to this lap
    """

    race_id: str
    lap: int
    positions: dict[str, int]
    gaps: dict[str, float]
    tyres: dict[str, tuple[str, int]]
    safety_car: bool
    total_laps: int = 57
    circuit: str = ""
    status: str = "RACING"
    weather: dict[str, Any] = field(default_factory=dict)
    last_lap_times: dict[str, float] = field(default_factory=dict)
    pit_stops_count: dict[str, int] = field(default_factory=dict)
    intervals: dict[str, float] = field(default_factory=dict)
    retired: dict[str, str] = field(default_factory=dict)
    fastest_lap: dict[str, Any] = field(default_factory=dict)
    pit_stops_history: list[dict[str, Any]] = field(default_factory=list)
    race_control_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to JSON-serializable dictionary."""
        return {
            "race_id": self.race_id,
            "lap": self.lap,
            "positions": self.positions,
            "gaps": self.gaps,
            "intervals": self.intervals,
            "tyres": {d: [comp, age] for d, (comp, age) in self.tyres.items()},
            "safety_car": self.safety_car,
            "total_laps": self.total_laps,
            "circuit": self.circuit,
            "status": self.status,
            "weather": self.weather,
            "last_lap_times": self.last_lap_times,
            "pit_stops_count": self.pit_stops_count,
            "retired": self.retired,
            "fastest_lap": self.fastest_lap,
            "pit_stops_history": self.pit_stops_history,
            "race_control_events": self.race_control_events,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RaceState:
        """Instantiate RaceState from a dictionary."""
        tyres = {
            d: (val[0], int(val[1])) if isinstance(val, (list, tuple)) else (str(val), 0)
            for d, val in data.get("tyres", {}).items()
        }
        return cls(
            race_id=data.get("race_id", ""),
            lap=int(data.get("lap", 1)),
            positions={d: int(pos) for d, pos in data.get("positions", {}).items()},
            gaps={d: float(gap) for d, gap in data.get("gaps", {}).items()},
            intervals={d: float(itv) for d, itv in data.get("intervals", {}).items()},
            tyres=tyres,
            safety_car=bool(data.get("safety_car", False)),
            total_laps=int(data.get("total_laps", 57)),
            circuit=data.get("circuit", ""),
            status=data.get("status", "RACING"),
            weather=data.get("weather", {}),
            last_lap_times={d: float(t) for d, t in data.get("last_lap_times", {}).items() if t is not None},
            pit_stops_count={d: int(c) for d, c in data.get("pit_stops_count", {}).items()},
            retired={d: str(r) for d, r in data.get("retired", {}).items()},
            fastest_lap=data.get("fastest_lap", {}),
            pit_stops_history=data.get("pit_stops_history", []),
            race_control_events=data.get("race_control_events", []),
        )
