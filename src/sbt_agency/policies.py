"""Baseline policies for the ring agent environment."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def cost_map_from_config(config, action_names: Sequence[str]) -> dict[str, int]:
    """Build a cost map consistent with the provided action names."""
    mapping = {
        "LEFT": config.cost_left,
        "RIGHT": config.cost_right,
        "REPAIR": config.cost_repair,
        "LEARN": config.cost_learn,
    }
    return {name: int(mapping[name]) for name in action_names if name in mapping}


def _feasible_actions(action_names: Sequence[str], cost_by_name: dict[str, int], r: int) -> list[int]:
    feasible = []
    for idx, name in enumerate(action_names):
        cost = cost_by_name.get(name, 0)
        if cost <= r:
            feasible.append(idx)
    return feasible


def make_random_feasible(
    action_names: Sequence[str],
    cost_by_name: dict[str, int],
    rng: np.random.Generator,
) -> Callable[[tuple, int], int]:
    """Uniform random policy over feasible actions."""

    def pi(state_tuple: tuple, _t: int) -> int:
        r = int(state_tuple[3])
        feasible = _feasible_actions(action_names, cost_by_name, r)
        if feasible:
            return int(rng.choice(feasible))
        if "LEFT" in action_names:
            return action_names.index("LEFT")
        return 0

    return pi


def make_maintenance_first(
    action_names: Sequence[str],
    cost_by_name: dict[str, int],
) -> Callable[[tuple, int], int]:
    """Prefer repair when damaged, else move right if feasible."""

    def pi(state_tuple: tuple, _t: int) -> int:
        u = int(state_tuple[1])
        r = int(state_tuple[3])
        if "REPAIR" in action_names and u == 1:
            if cost_by_name.get("REPAIR", 0) <= r:
                return action_names.index("REPAIR")

        if "RIGHT" in action_names and cost_by_name.get("RIGHT", 0) <= r:
            return action_names.index("RIGHT")

        feasible = _feasible_actions(action_names, cost_by_name, r)
        if "LEFT" in action_names and cost_by_name.get("LEFT", 0) <= r:
            return action_names.index("LEFT")
        if feasible:
            return feasible[0]
        return 0

    return pi


def make_move_right_if_possible(
    action_names: Sequence[str],
    cost_by_name: dict[str, int],
    rng: np.random.Generator | None = None,
) -> Callable[[tuple, int], int]:
    """Move right when feasible; otherwise pick a fallback action."""

    def pi(state_tuple: tuple, _t: int) -> int:
        r = int(state_tuple[3])
        if "RIGHT" in action_names and cost_by_name.get("RIGHT", 0) <= r:
            return action_names.index("RIGHT")

        feasible = _feasible_actions(action_names, cost_by_name, r)
        if rng is not None and feasible:
            return int(rng.choice(feasible))
        if "LEFT" in action_names and cost_by_name.get("LEFT", 0) <= r:
            return action_names.index("LEFT")
        if feasible:
            return feasible[0]
        return 0

    return pi

