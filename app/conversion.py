"""Packaging-unit conversion graph with exact rational arithmetic.

An equation such as ``2箱=24件`` attached to a supplier record means
``1箱 = 12件``.  Equations are undirected edges whose rational rate is
inverted when traversed backwards, so reverse and multi-hop conversions are
the same graph traversal.  All arithmetic uses :class:`fractions.Fraction`,
which keeps results exact and lets callers render the simplest fraction.

Every connected component of the graph must keep a single consistent rate
between any two units; :func:`find_rate_conflict` returns the first
contradicting cycle (the two paths disagree), which is what the UI reports
when adding an equation or merging two supplier groups.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

UNIT_PIECE = "件"


@dataclass(frozen=True)
class Equation:
    id: int
    record_id: int
    unit_a: str
    qty_a: int
    unit_b: str
    qty_b: int
    created_at: str | None = None


def normalize_unit(value: object) -> str:
    """Trim and validate a unit name. Raises ValueError on bad input."""
    if not isinstance(value, str):
        raise ValueError("单位名必须是文本")
    unit = value.strip()
    if not unit:
        raise ValueError("单位名不能为空")
    if len(unit) > 20:
        raise ValueError("单位名最长 20 个字符")
    return unit


def parse_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是正整数")
    if isinstance(value, int):
        amount = value
    else:
        raise ValueError(f"{label}必须是正整数")
    if amount <= 0:
        raise ValueError(f"{label}必须是正整数")
    return amount


def edge_rate(equation: Mapping[str, object]) -> Fraction:
    """Rate in 件-style units: value of one ``unit_a`` measured in ``unit_b``."""
    return Fraction(int(equation["qty_b"]), int(equation["qty_a"]))  # type: ignore[arg-type]


def canonical_rate(
    unit_a: str, qty_a: int, unit_b: str, qty_b: int
) -> tuple[tuple[str, str], Fraction]:
    """Orientation-independent identity of an equation.

    Returns the unit pair sorted by name and the rate of the first sorted unit
    measured in the second, so ``2箱=24件`` and ``24件=2箱`` compare equal.
    """
    if unit_a < unit_b:
        return (unit_a, unit_b), Fraction(qty_b, qty_a)
    return (unit_b, unit_a), Fraction(qty_a, qty_b)


def build_adjacency(
    equations: Sequence[Mapping[str, object]],
) -> dict[str, list[tuple[str, Fraction, Mapping[str, object]]]]:
    adjacency: dict[str, list[tuple[str, Fraction, Mapping[str, object]]]] = {}
    for equation in equations:
        unit_a = str(equation["unit_a"])
        unit_b = str(equation["unit_b"])
        rate = edge_rate(equation)
        adjacency.setdefault(unit_a, []).append((unit_b, rate, equation))
        adjacency.setdefault(unit_b, []).append((unit_a, 1 / rate, equation))
    return adjacency


@dataclass(frozen=True)
class ConversionPath:
    rate: Fraction
    edges: tuple[Mapping[str, object], ...]


def find_conversion(
    adjacency: Mapping[str, list[tuple[str, Fraction, Mapping[str, object]]]],
    source: str,
    target: str,
) -> ConversionPath | None:
    """BFS for the shortest hop path with the exact source->target rate."""
    if source == target:
        return ConversionPath(rate=Fraction(1), edges=())
    if source not in adjacency:
        return None
    queue: deque[str] = deque([source])
    rates: dict[str, Fraction] = {source: Fraction(1)}
    previous: dict[str, tuple[str, Mapping[str, object]] | None] = {source: None}
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for neighbour, edge_rate_value, equation in adjacency.get(node, []):
            if neighbour in rates:
                continue
            rates[neighbour] = rates[node] * edge_rate_value
            previous[neighbour] = (node, equation)
            queue.append(neighbour)
    if target not in rates:
        return None
    edges: list[Mapping[str, object]] = []
    cursor = target
    while previous[cursor] is not None:
        parent, equation = previous[cursor]  # type: ignore[misc]
        edges.append(equation)
        cursor = parent
    edges.reverse()
    return ConversionPath(rate=rates[target], edges=tuple(edges))


def find_rate_conflict(
    equations: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Return the first inconsistent cycle found, else ``None``.

    Edges are processed in the given order, so callers sort equations by id
    (candidate equation last) to make the candidate part of every reported
    conflict.
    """
    rates: dict[str, Fraction] = {}
    previous: dict[str, tuple[str, Mapping[str, object]] | None] = {}
    adjacency = build_adjacency(equations)

    def walk_back(node: str) -> list[Mapping[str, object]]:
        path: list[Mapping[str, object]] = []
        cursor = node
        while previous.get(cursor) is not None:
            parent, equation = previous[cursor]  # type: ignore[misc]
            path.append(equation)
            cursor = parent
        path.reverse()
        return path

    for equation in equations:
        unit_a, unit_b = str(equation["unit_a"]), str(equation["unit_b"])
        rate = edge_rate(equation)
        known_a = unit_a in rates
        known_b = unit_b in rates
        if known_a and known_b:
            expected = rates[unit_b] / rates[unit_a]
            if expected != rate:
                path_a = walk_back(unit_a)
                path_b = walk_back(unit_b)
                return {
                    "unit_a": unit_a,
                    "unit_b": unit_b,
                    "candidate": equation,
                    "path_a": path_a,
                    "path_b": path_b,
                    "existing_rate_b_over_a": expected,
                    "candidate_rate_b_over_a": rate,
                }
            continue
        if not known_a and not known_b:
            rates[unit_a] = Fraction(1)
            previous[unit_a] = None
            rates[unit_b] = rates[unit_a] * rate
            previous[unit_b] = (unit_a, equation)
        elif not known_a:
            # unit_b known: derive unit_a from the edge rate.
            rates[unit_a] = rates[unit_b] / rate
            previous[unit_a] = (unit_b, equation)
        else:
            rates[unit_b] = rates[unit_a] * rate
            previous[unit_b] = (unit_a, equation)

    # Defensive pass: if the graph contains a component the ordered walk above
    # did not root (cannot normally happen), verify every edge explicitly.
    for equation in equations:
        unit_a, unit_b = str(equation["unit_a"]), str(equation["unit_b"])
        if unit_a in rates and unit_b in rates:
            if rates[unit_b] / rates[unit_a] != edge_rate(equation):
                return {
                    "unit_a": unit_a,
                    "unit_b": unit_b,
                    "candidate": equation,
                    "path_a": walk_back(unit_a),
                    "path_b": walk_back(unit_b),
                    "existing_rate_b_over_a": rates[unit_b] / rates[unit_a],
                    "candidate_rate_b_over_a": edge_rate(equation),
                }
    return None


def reachable_units(
    adjacency: Mapping[str, list[tuple[str, Fraction, Mapping[str, object]]]],
    start: str,
) -> list[str]:
    if start not in adjacency:
        return []
    seen = {start}
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        for neighbour, _, _ in adjacency.get(node, []):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return sorted(seen)


def group_equation_conflicts(
    equations_by_group: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Invariant check used at startup: every inconsistent group is reported."""
    conflicts: list[dict[str, object]] = []
    for group_key in sorted(equations_by_group):
        conflict = find_rate_conflict(
            sorted(equations_by_group[group_key], key=lambda item: int(item["id"]))
        )
        if conflict is not None:
            conflict["group_key"] = group_key
            conflicts.append(conflict)
    return conflicts


def fraction_text(rate: Fraction) -> str:
    """Human rendering of the simplest fraction: integer or ``a/b``."""
    if rate.denominator == 1:
        return str(rate.numerator)
    return f"{rate.numerator}/{rate.denominator}"
