from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..protocol import Action, Observation


class Agent:
    """One method matters: `act(observation) -> action`.

    Agents are built fresh for every game with their seat and a private seed, so a
    game is a pure function of (seed, agent specs).
    """

    name = "agent"

    def __init__(self, seat: int, seed: int):
        self.seat = seat
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, Any]:
        """Agent-side records read by the harness after the game (e.g. beliefs)."""
        return {}


@dataclass(frozen=True)
class AgentSpec:
    """A picklable recipe for building an agent, so games can run in worker processes."""

    cls: type[Agent]
    kwargs: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or self.cls.name

    def build(self, seat: int, seed: int) -> Agent:
        agent = self.cls(seat, seed, **self.kwargs)
        agent.name = self.name
        return agent
