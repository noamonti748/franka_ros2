"""Generic stabilization for the autonomous policy's learned task mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Phase


@dataclass(frozen=True)
class ModeDecision:
    probabilities: np.ndarray
    phase: Phase
    changed: bool
    confidence: float
    advance_prob: float


class AutonomousModeFilter:
    """Advance using learned mode + optional advance_prob (no geometric gates)."""

    def __init__(
        self,
        *,
        alpha: float = 0.35,
        margin: float = 0.15,
        consecutive_steps: int = 3,
        advance_threshold: float = 0.5,
        min_dwell_steps: int = 20,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("mode alpha must be in (0, 1]")
        if margin < 0.0 or consecutive_steps <= 0:
            raise ValueError("mode margin and consecutive steps are invalid")
        if min_dwell_steps < 0:
            raise ValueError("min_dwell_steps must be nonnegative")
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.consecutive_steps = int(consecutive_steps)
        self.advance_threshold = float(advance_threshold)
        self.min_dwell_steps = int(min_dwell_steps)
        self.reset()

    def reset(self, phase: Phase = Phase.HOVER_RED) -> None:
        self.phase = Phase(phase)
        self.probabilities = np.zeros(8, dtype=np.float32)
        self.probabilities[int(self.phase)] = 1.0
        self._candidate: Phase | None = None
        self._candidate_steps = 0
        self.phase_step = 0

    def force(self, phase: Phase) -> None:
        self.reset(phase)

    def update(
        self,
        raw_probabilities: np.ndarray,
        advance_prob: float | None = None,
    ) -> ModeDecision:
        raw = np.asarray(raw_probabilities, dtype=np.float32)
        if raw.shape != (8,) or not np.all(np.isfinite(raw)):
            raise ValueError("mode probabilities must be finite with shape (8,)")
        raw = np.maximum(raw, 0.0)
        total = float(np.sum(raw))
        if total <= 0.0:
            raise ValueError("mode probabilities have no positive mass")
        raw /= total
        filtered = (1.0 - self.alpha) * self.probabilities + self.alpha * raw
        filtered /= max(float(np.sum(filtered)), 1.0e-8)
        self.probabilities = filtered.astype(np.float32)

        current = int(self.phase)
        next_phase = min(current + 1, int(Phase.PLACED))
        candidate = Phase(int(np.argmax(self.probabilities)))
        advance_value = (
            1.0 if advance_prob is None else float(advance_prob)
        )
        if advance_prob is not None and self.advance_threshold >= 0.0:
            # Advance head is the sole transition driver: step to next_phase
            # monotonically when it fires, mirroring GATE_MODE_LEARNED in sim.
            eligible = bool(
                current < int(Phase.PLACED)
                and self.phase_step >= self.min_dwell_steps
                and advance_value >= self.advance_threshold
            )
            changed = eligible
            if changed:
                self.phase = Phase(next_phase)
                self.phase_step = 0
                self._candidate = None
                self._candidate_steps = 0
            else:
                self.phase_step += 1
        else:
            eligible = bool(
                int(candidate) == next_phase
                and self.probabilities[next_phase]
                >= self.probabilities[current] + self.margin
            )
            if eligible:
                if self._candidate == candidate:
                    self._candidate_steps += 1
                else:
                    self._candidate = candidate
                    self._candidate_steps = 1
            else:
                self._candidate = None
                self._candidate_steps = 0
            changed = self._candidate_steps >= self.consecutive_steps
            if changed:
                self.phase = candidate
                self.phase_step = 0
                self._candidate = None
                self._candidate_steps = 0
            else:
                self.phase_step += 1
        return ModeDecision(
            probabilities=self.probabilities.copy(),
            phase=self.phase,
            changed=changed,
            confidence=float(self.probabilities[int(self.phase)]),
            advance_prob=advance_value,
        )
