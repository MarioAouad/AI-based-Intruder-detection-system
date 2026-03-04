"""
threat_timer.py  –  Temporal Logic & Per-ID Zone Memory
=========================================================

ZoneTimer tracks every ByteTrack ID independently.  It answers one question:

    "Has this person been inside the 2-metre zone for 30 continuous seconds?"

State machine per ID
---------------------
  NOT IN ZONE  →  person detected outside zone / not yet seen
  IN ZONE      →  person inside zone; countdown running
  TRIGGERED    →  30 s elapsed; caller must act and call reset()

The timer resets immediately if the person leaves the zone or disappears
from the frame, enforcing the "30 continuous seconds" rule strictly.
"""

import time
from typing import Dict, Tuple

from config import ZONE_RADIUS_METERS, TRIGGER_TIME_SECONDS


class ZoneTimer:
    """
    Maintains independent countdown timers for each tracked person.

    Internal state
    --------------
    _timers : dict { track_id (int) → start_time (float, UNIX epoch) }
        Stores the moment a given ID first entered the zone in this
        current uninterrupted visit.  Missing key = person is outside zone.

    Usage
    -----
    timer = ZoneTimer()
    triggered, elapsed = timer.update(track_id=3, distance_meters=1.5)
    if triggered:
        timer.reset(track_id=3)   # mandatory after acting on a trigger
    """

    def __init__(
        self,
        zone_radius_m: float        = ZONE_RADIUS_METERS,
        trigger_time_s: float       = TRIGGER_TIME_SECONDS,
    ):
        self.zone_radius_m  = zone_radius_m
        self.trigger_time_s = trigger_time_s

        # { track_id: unix_timestamp_of_zone_entry }
        self._timers: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def update(
        self,
        track_id: int,
        distance_m: float | None,
    ) -> Tuple[bool, float]:
        """
        Called once per frame for every detected person.

        Parameters
        ----------
        track_id   : ByteTrack integer ID for this person.
        distance_m : Current distance in metres, or None if uncalibrated.

        Returns
        -------
        triggered : bool
            True if the person has been inside the zone for ≥ trigger_time_s
            continuously.  The caller MUST call reset(track_id) after acting.
        elapsed_s : float
            Seconds this ID has been inside the zone (0.0 if outside / None).
        """
        # If distance is unknown (uncalibrated), treat as 'outside zone'.
        in_zone = (distance_m is not None) and (distance_m <= self.zone_radius_m)

        if in_zone:
            if track_id not in self._timers:
                # ── New entry: start the clock ─────────────────────────
                self._timers[track_id] = time.monotonic()

            elapsed  = time.monotonic() - self._timers[track_id]
            triggered = elapsed >= self.trigger_time_s
            return triggered, round(elapsed, 1)

        else:
            # ── Person left the zone or unknown distance ─────────────────
            # Hard-reset: they must spend another full 30 s continuously.
            if track_id in self._timers:
                del self._timers[track_id]
            return False, 0.0

    def reset(self, track_id: int) -> None:
        """
        Force-reset a specific ID's timer.
        Call this immediately after processing a TRIGGER to avoid
        re-triggering every frame until the person leaves.
        """
        if track_id in self._timers:
            del self._timers[track_id]

    def remove(self, track_id: int) -> None:
        """
        Remove an ID that has disappeared from the frame entirely
        (called by the orchestrator when an ID is no longer detected).
        Alias for reset() — kept as a semantic alias for clarity.
        """
        self.reset(track_id)

    def purge_stale(self, active_ids: set) -> None:
        """
        Remove timer entries for IDs that have vanished from the current frame.
        Call once per frame after processing all detections.

        Parameters
        ----------
        active_ids : set of int
            Track IDs that were detected in the current frame.
        """
        stale = [tid for tid in self._timers if tid not in active_ids]
        for tid in stale:
            del self._timers[tid]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def elapsed(self, track_id: int) -> float:
        """Return seconds in zone for an ID, or 0.0 if not being tracked."""
        if track_id not in self._timers:
            return 0.0
        return round(time.monotonic() - self._timers[track_id], 1)

    def active_count(self) -> int:
        """How many unique IDs currently have active timers."""
        return len(self._timers)

    def __repr__(self) -> str:
        active = {tid: round(time.monotonic() - ts, 1) for tid, ts in self._timers.items()}
        return f"ZoneTimer(zone={self.zone_radius_m}m, trigger={self.trigger_time_s}s, active={active})"
