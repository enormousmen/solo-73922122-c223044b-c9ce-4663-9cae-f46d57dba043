"""Conversion workflow: equation CRUD, group-scoped queries, conflict payloads.

All writes happen inside a single ``BEGIN IMMEDIATE`` transaction and are
validated against the latest committed state inside that transaction, so the
two Web processes can never each pass validation and then jointly commit a
contradiction.  Reads (including a conversion query) evaluate grouping,
equations and the result inside one read transaction, i.e. one committed
snapshot.
"""

from __future__ import annotations

import hashlib
import sqlite3
from fractions import Fraction
from typing import Any, Mapping, Sequence

from . import conversion, db
from .grouping import Grouping, build_grouping, canonical_json

MAX_EQUATIONS_PER_RECORD = 20


class ApiProblem(Exception):
    """Re-exported alias so callers only import one service module."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra

    def body(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {"code": self.code, "message": self.message, **self.extra},
        }


# --------------------------------------------------------------------- loading


def load_records(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, seq, source, source_code, name FROM records ORDER BY id"
    ).fetchall()
    return [dict(row) for row in rows]


def load_decisions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, record_a, record_b, relation, created_at "
        "FROM decisions ORDER BY id"
    ).fetchall()
    return [dict(row) for row in rows]


def load_equations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, record_id, unit_a, qty_a, unit_b, qty_b, created_at "
        "FROM unit_equations ORDER BY id"
    ).fetchall()
    return [dict(row) for row in rows]


def current_grouping(
    conn: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Grouping]:
    records = load_records(conn)
    decisions = load_decisions(conn)
    grouping = build_grouping([int(r["id"]) for r in records], decisions)
    return records, decisions, grouping


# --------------------------------------------------------------------- views


def record_brief(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": int(record["id"]),
        "source": record["source"],
        "source_code": record["source_code"],
        "name": record["name"],
    }


def equation_text(equation: Mapping[str, Any]) -> str:
    return (
        f"{int(equation['qty_a'])}{equation['unit_a']}"
        f"={int(equation['qty_b'])}{equation['unit_b']}"
    )


def equation_view(
    equation: Mapping[str, Any],
    by_id: Mapping[int, Mapping[str, Any]] | None = None,
    group_key: str | None = None,
) -> dict[str, Any]:
    record_id = int(equation["record_id"])
    payload = {
        "id": int(equation["id"]),
        "record_id": record_id,
        "record": record_brief(by_id[record_id]) if by_id and record_id in by_id else None,
        "group_key": group_key,
        "unit_a": equation["unit_a"],
        "qty_a": int(equation["qty_a"]),
        "unit_b": equation["unit_b"],
        "qty_b": int(equation["qty_b"]),
        "text": equation_text(equation),
        "created_at": equation.get("created_at"),
    }
    return payload


def equations_by_group(
    grouping: Grouping, equations: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in grouping.by_key}
    for equation in equations:
        key = grouping.group_of.get(int(equation["record_id"]))
        if key is not None:
            grouped.setdefault(key, []).append(dict(equation))
    return grouped


def group_units(equations: Sequence[Mapping[str, Any]]) -> list[str]:
    units: set[str] = set()
    for equation in equations:
        units.add(str(equation["unit_a"]))
        units.add(str(equation["unit_b"]))
    return sorted(units)


# --------------------------------------------------------------- state version


def compute_state_version(
    grouping: Grouping,
    decisions: Sequence[Mapping[str, Any]],
    equations: Sequence[Mapping[str, Any]],
) -> str:
    """Digest of everything a conversion result can depend on."""
    snapshot = {
        "groups": {
            key: sorted(ids) for key, ids in sorted(grouping.by_key.items())
        },
        "decisions": [
            {
                "id": int(d["id"]),
                "record_a": int(d["record_a"]),
                "record_b": int(d["record_b"]),
                "relation": d["relation"],
            }
            for d in decisions
        ],
        "equations": [
            {
                "id": int(e["id"]),
                "record_id": int(e["record_id"]),
                "unit_a": e["unit_a"],
                "qty_a": int(e["qty_a"]),
                "unit_b": e["unit_b"],
                "qty_b": int(e["qty_b"]),
            }
            for e in equations
        ],
    }
    return hashlib.sha256(
        canonical_json(snapshot).encode("utf-8")
    ).hexdigest()[:20]


# ------------------------------------------------------------------- conflicts


def _rate_line(rate: Fraction, unit_a: str, unit_b: str) -> str:
    return f"1{unit_a} = {conversion.fraction_text(rate)}{unit_b}"


def equation_conflict_payload(
    conflict: Mapping[str, Any],
    by_id: Mapping[int, Mapping[str, Any]],
    group_key: str | None = None,
) -> dict[str, Any]:
    candidate = conflict["candidate"]
    seen: set[int] = set()
    evidence: list[dict[str, Any]] = []
    for equation in list(conflict["path_a"]) + list(conflict["path_b"]):
        eq_id = int(equation["id"])
        if eq_id in seen:
            continue
        seen.add(eq_id)
        evidence.append(equation_view(equation, by_id, group_key))

    unit_a, unit_b = conflict["unit_a"], conflict["unit_b"]
    existing_rate = conflict["existing_rate_b_over_a"]
    candidate_rate = conflict["candidate_rate_b_over_a"]
    message = (
        f"组内换算闭环比例矛盾：现有通路给出 {_rate_line(existing_rate, unit_a, unit_b)}，"
        f"而等式“{equation_text(candidate)}”给出 {_rate_line(candidate_rate, unit_a, unit_b)}。"
        "同一组内相同两个单位之间的换算比例必须一致，已拒绝该操作。"
    )
    return {
        "code": "unit_ratio_conflict",
        "message": message,
        "group_key": group_key,
        "units": [unit_a, unit_b],
        "candidate": equation_view(candidate, by_id, group_key),
        "evidence": evidence,
        "existing_rate": conversion.fraction_text(existing_rate),
        "candidate_rate": conversion.fraction_text(candidate_rate),
    }


# --------------------------------------------------------------------- writes


def _candidate_equation(
    record_id: int, unit_a: str, qty_a: int, unit_b: str, qty_b: int
) -> dict[str, Any]:
    # Sentinel id larger than any real row id keeps the candidate last in the
    # id ordering used by the conflict detector.
    return {
        "id": 10**12,
        "record_id": record_id,
        "unit_a": unit_a,
        "qty_a": qty_a,
        "unit_b": unit_b,
        "qty_b": qty_b,
        "created_at": None,
    }


def add_equation(
    conn: sqlite3.Connection,
    record_id: object,
    raw_unit_a: object,
    raw_qty_a: object,
    raw_unit_b: object,
    raw_qty_b: object,
) -> dict[str, Any]:
    if not isinstance(record_id, int) or isinstance(record_id, bool):
        raise ApiProblem(400, "invalid_record", "record_id 必须是整数")
    try:
        unit_a = conversion.normalize_unit(raw_unit_a)
        unit_b = conversion.normalize_unit(raw_unit_b)
        qty_a = conversion.parse_positive_int(raw_qty_a, "数量")
        qty_b = conversion.parse_positive_int(raw_qty_b, "数量")
    except ValueError as exc:
        raise ApiProblem(400, "invalid_equation", str(exc)) from exc
    if unit_a == unit_b:
        raise ApiProblem(
            400, "same_unit", "等式两端的单位名不能相同，例如应填写 2箱=24件。"
        )

    with db.write_tx(conn):
        record = conn.execute(
            "SELECT id, source, source_code, name FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()
        if record is None:
            raise ApiProblem(404, "unknown_record", "供应商原始记录不存在")

        count = conn.execute(
            "SELECT COUNT(*) AS count FROM unit_equations WHERE record_id = ?",
            (record_id,),
        ).fetchone()["count"]
        if int(count) >= MAX_EQUATIONS_PER_RECORD:
            raise ApiProblem(
                409,
                "equation_limit_reached",
                f"每条供应商原始记录最多 {MAX_EQUATIONS_PER_RECORD} 条换算等式，"
                "请先删除旧等式再新增。",
            )

        records, decisions, grouping = current_grouping(conn)
        by_id = {int(row["id"]): row for row in records}
        equations = load_equations(conn)

        # A restatement of the same rate on the same unit pair counts as a
        # duplicate, regardless of orientation (2箱=24件 vs 24件=2箱).
        candidate_pair, candidate_rate = conversion.canonical_rate(
            unit_a, qty_a, unit_b, qty_b
        )
        duplicate = next(
            (
                equation
                for equation in equations
                if int(equation["record_id"]) == record_id
                and conversion.canonical_rate(
                    str(equation["unit_a"]),
                    int(equation["qty_a"]),
                    str(equation["unit_b"]),
                    int(equation["qty_b"]),
                )
                == (candidate_pair, candidate_rate)
            ),
            None,
        )
        if duplicate is not None:
            raise ApiProblem(
                409,
                "equation_exists",
                f"该记录已存在等价等式“{equation_text(duplicate)}”，请勿重复添加。",
                existing=equation_view(duplicate, by_id),
            )

        group_key = grouping.group_of[record_id]
        group_equations = [
            dict(equation)
            for equation in equations
            if grouping.group_of.get(int(equation["record_id"])) == group_key
        ]
        candidate = _candidate_equation(record_id, unit_a, qty_a, unit_b, qty_b)
        ordered = sorted(group_equations, key=lambda item: int(item["id"])) + [candidate]
        conflict = conversion.find_rate_conflict(ordered)
        if conflict is not None:
            raise ApiProblem(
                409,
                "unit_ratio_conflict",
                equation_conflict_payload(conflict, by_id, group_key)["message"],
                conflict=equation_conflict_payload(conflict, by_id, group_key),
            )

        from .service import utc_now  # local import avoids a module cycle

        created_at = utc_now()
        cursor = conn.execute(
            "INSERT INTO unit_equations "
            "(record_id, unit_a, qty_a, unit_b, qty_b, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (record_id, unit_a, qty_a, unit_b, qty_b, created_at),
        )
        new_row = conn.execute(
            "SELECT id, record_id, unit_a, qty_a, unit_b, qty_b, created_at "
            "FROM unit_equations WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        final_equations = load_equations(conn)
        return {
            "ok": True,
            "equation": equation_view(dict(new_row), by_id, group_key),
            "state_version": compute_state_version(
                grouping, decisions, final_equations
            ),
        }


def delete_equation(conn: sqlite3.Connection, equation_id: object) -> dict[str, Any]:
    if not isinstance(equation_id, int) or isinstance(equation_id, bool):
        raise ApiProblem(400, "invalid_equation_id", "equation_id 必须是整数")
    with db.write_tx(conn):
        row = conn.execute(
            "SELECT id FROM unit_equations WHERE id = ?", (equation_id,)
        ).fetchone()
        if row is None:
            raise ApiProblem(404, "unknown_equation", "要删除的换算等式不存在")
        conn.execute("DELETE FROM unit_equations WHERE id = ?", (equation_id,))
        return {"ok": True, "deleted_id": equation_id}


# ---------------------------------------------------------------------- query


def query_conversion(
    conn: sqlite3.Connection,
    record_id: object,
    raw_unit: object,
    raw_quantity: object,
) -> dict[str, Any]:
    if not isinstance(record_id, int) or isinstance(record_id, bool):
        raise ApiProblem(400, "invalid_record", "record_id 必须是整数")
    try:
        unit = conversion.normalize_unit(raw_unit)
        quantity = conversion.parse_positive_int(raw_quantity, "采购量")
    except ValueError as exc:
        raise ApiProblem(400, "invalid_query", str(exc)) from exc

    with db.read_tx(conn):
        record = conn.execute(
            "SELECT id, source, source_code, name FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()
        if record is None:
            raise ApiProblem(404, "unknown_record", "供应商原始记录不存在")

        records, decisions, grouping = current_grouping(conn)
        by_id = {int(row["id"]): row for row in records}
        equations = load_equations(conn)
        group_key = grouping.group_of[record_id]
        group_equations = [
            dict(equation)
            for equation in equations
            if grouping.group_of.get(int(equation["record_id"])) == group_key
        ]
        adjacency = conversion.build_adjacency(group_equations)
        version = compute_state_version(grouping, decisions, equations)

        base = {
            "record": record_brief(dict(record)),
            "group_key": group_key,
            "unit": unit,
            "quantity": quantity,
            "target_unit": conversion.UNIT_PIECE,
            "state_version": version,
            "group_units": group_units(group_equations),
        }

        path = conversion.find_conversion(
            adjacency, unit, conversion.UNIT_PIECE
        )
        if path is None:
            reachable = conversion.reachable_units(adjacency, unit)
            if reachable:
                hint = (
                    f"组内没有从“{unit}”到“{conversion.UNIT_PIECE}”的换算通路；"
                    f"从“{unit}”出发只能到达：{'、'.join(reachable)}。"
                    "请在本组任一供应商记录上补充缺失的换算等式。"
                )
            else:
                hint = (
                    f"本组没有任何包含单位“{unit}”的换算等式，"
                    f"无法换算为“{conversion.UNIT_PIECE}”，请先补充等式。"
                )
            return {
                "ok": True,
                "query": {
                    **base,
                    "reachable": False,
                    "message": hint,
                    "reachable_units": reachable,
                }
            }

        rate = path.rate
        result = Fraction(quantity) * rate
        used_equations = [
            equation_view(equation, by_id, group_key) for equation in path.edges
        ]
        return {
            "ok": True,
            "query": {
                **base,
                "reachable": True,
                "rate_per_unit": conversion.fraction_text(rate),
                "result_numerator": result.numerator,
                "result_denominator": result.denominator,
                "result_text": conversion.fraction_text(result),
                "is_integer": result.denominator == 1,
                "hops": len(path.edges),
                "used_equations": used_equations,
                "expression": _expression(quantity, unit, result, used_equations),
            },
        }


def _expression(
    quantity: int,
    unit: str,
    result: Fraction,
    used_equations: Sequence[Mapping[str, Any]],
) -> str:
    chain = " → ".join(
        f"{equation['text']}（#{equation['id']}）" for equation in used_equations
    )
    return (
        f"{quantity}{unit}，经 {len(used_equations)} 跳换算 [{chain}]，"
        f"= {conversion.fraction_text(result)}{conversion.UNIT_PIECE}"
    )


# ----------------------------------------------------------- merge integration


def prospective_merge_conflict(
    grouping: Grouping,
    equations: Sequence[Mapping[str, Any]],
    involved_records: Sequence[int],
    by_id: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Conflict that would appear if the given records ended up in one group.

    Used both by the relation-merge preview and by confirm-time revalidation.
    """
    group_keys = {grouping.group_of[rid] for rid in involved_records}
    union: list[dict[str, Any]] = []
    for equation in equations:
        if grouping.group_of.get(int(equation["record_id"])) in group_keys:
            union.append(dict(equation))
    if not union:
        return None
    ordered = sorted(union, key=lambda item: int(item["id"]))
    conflict = conversion.find_rate_conflict(ordered)
    if conflict is None:
        return None
    merged_key = next(iter(group_keys))
    return equation_conflict_payload(conflict, by_id, merged_key)
