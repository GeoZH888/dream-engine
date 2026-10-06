"""Dream bizarreness operators (Hobson: discontinuity, incongruity, uncertainty).

Each operator is a pure function ``(seed, rng, params) -> seed | None`` that adds one
``OperatorApplication``: a concrete directive the narrator must enact, naming the
fragments and entities it acts on. ``None`` means the operator does not apply to
this seed (e.g. ``identity_merge`` needs two people).
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import timedelta
from typing import Callable

import numpy as np

from dream_engine.seed import DreamSeed, OperatorApplication

Operator = Callable[[DreamSeed, np.random.Generator, dict], "DreamSeed | None"]

PHYSICS_VIOLATIONS = [
    "the dreamer can fly, or float just above the ground",
    "the dreamer breathes underwater without effort",
    "gravity shifts sideways and people walk on the walls",
    "a room is much larger on the inside than the outside",
    "the dreamer walks through a solid wall",
    "falling continues far longer than any height allows",
]
OBJECT_TARGETS = ["an animal", "a liquid", "a person", "a building", "light", "a letter or a book",
                  "a musical instrument", "food"]
AGE_SHIFTS = ["is a child again", "is much older", "is a teenager at school"]


def _pick(items: list, rng: np.random.Generator):
    return items[int(rng.integers(len(items)))]


def _two(items: list, rng: np.random.Generator) -> tuple:
    i, j = rng.choice(len(items), size=2, replace=False)
    return items[int(i)], items[int(j)]


def _add(seed: DreamSeed, op: OperatorApplication) -> DreamSeed:
    return replace(seed, operators=seed.operators + (op,))


# ----------------------------------------------------------------------------- operators

def identity_merge(seed: DreamSeed, rng: np.random.Generator, params: dict) -> DreamSeed | None:
    people = seed.entities_of("person")
    if len(people) < 2:
        return None
    (a, fa), (b, fb) = _two(people, rng)
    return _add(seed, OperatorApplication(
        "identity_merge", f"{a} and {b} are one and the same person (\"{a}, who was also {b}\").", (fa, fb, a, b)))


def place_fusion(seed: DreamSeed, rng: np.random.Generator, params: dict) -> DreamSeed | None:
    places = seed.entities_of("place")
    if len(places) < 2:
        return None
    (a, fa), (b, fb) = _two(places, rng)
    return _add(seed, OperatorApplication(
        "place_fusion", f"{a} and {b} are fused into a single place: one opens directly onto the other.",
        (fa, fb, a, b)))


def scene_jump(seed: DreamSeed, rng: np.random.Generator, params: dict) -> DreamSeed | None:
    if len(seed.fragments) < 2:
        return None
    a, b = _two(list(seed.fragments), rng)
    return _add(seed, OperatorApplication(
        "scene_jump", f"Cut abruptly from the scene of [{a.id}] to the scene of [{b.id}], with no transition "
                      "and no explanation.", (a.id, b.id)))


def physics_violation(seed: DreamSeed, rng: np.random.Generator, params: dict) -> DreamSeed | None:
    return _add(seed, OperatorApplication("physics_violation", f"Physics breaks: {_pick(PHYSICS_VIOLATIONS, rng)}."))


def object_transformation(seed: DreamSeed, rng: np.random.Generator, params: dict) -> DreamSeed | None:
    objects = seed.entities_of("object")
    if not objects:
        return None
    obj, fid = _pick(objects, rng)
    other = [o for o, _ in objects if o != obj]
    target = _pick(other, rng) if other and rng.random() < params["object_swap_prob"] else _pick(OBJECT_TARGETS, rng)
    return _add(seed, OperatorApplication(
        "object_transformation", f"Mid-scene, the {obj} turns into {target}.", (fid, obj)))


def time_distortion(seed: DreamSeed, rng: np.random.Generator, params: dict) -> DreamSeed | None:
    newest = max(f.timestamp for f in seed.fragments)
    old = [f for f in seed.fragments if newest - f.timestamp > timedelta(days=params["old_memory_days"])]
    if old:
        f = _pick(old, rng)
        return _add(seed, OperatorApplication(
            "time_distortion", f"Past and present overlap: the time of [{f.id}] is happening now, "
                               "at the same time as the recent scenes.", (f.id,)))
    return _add(seed, OperatorApplication("time_distortion", f"The dreamer {_pick(AGE_SHIFTS, rng)}."))


def uncertainty(seed: DreamSeed, rng: np.random.Generator, params: dict) -> DreamSeed | None:
    for kind, phrase in (("person", "someone I knew, maybe {}"), ("place", "somewhere like {}, but not quite")):
        ents = seed.entities_of(kind)
        if ents:
            e, fid = _pick(ents, rng)
            return _add(seed, OperatorApplication(
                "uncertainty", f"The identity of {e} stays vague: \"{phrase.format(e)}\".", (fid, e)))
    return _add(seed, OperatorApplication(
        "uncertainty", "Where the dreamer is stays vague and unresolved."))


OPERATORS: dict[str, Operator] = {
    "identity_merge": identity_merge,
    "place_fusion": place_fusion,
    "scene_jump": scene_jump,
    "physics_violation": physics_violation,
    "object_transformation": object_transformation,
    "time_distortion": time_distortion,
    "uncertainty": uncertainty,
}


def apply_operators(seed: DreamSeed, k: int, rng: np.random.Generator, params: dict) -> DreamSeed:
    """Apply k distinct applicable operators, in random order."""
    names = list(OPERATORS)
    rng.shuffle(names)
    for name in names:
        if len(seed.operators) >= k:
            break
        seed = OPERATORS[name](seed, rng, params) or seed
    return seed


def structural_bizarreness(seed: DreamSeed, params: dict) -> float:
    """A prior from the seed alone (operators + PGO bursts), in [0, 1).

    The rated score comes from ``eval/bizarreness.py``; this is what the
    mechanism *intends*, recorded for comparison.
    """
    load = len(seed.operators) + params["prior_pgo_weight"] * len(seed.pgo_events)
    return round(1.0 - math.exp(-load / params["prior_scale"]), 3)
