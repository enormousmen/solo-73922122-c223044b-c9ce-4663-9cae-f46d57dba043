"""Tests for packaging-unit equations, group-scoped exact conversion and the
interaction between equation maintenance and supplier group merges."""

from __future__ import annotations

import sqlite3
from fractions import Fraction
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import conversion, db
from app.main import create_app

SEED = """来源,源编号,企业名称
采购系统A,A-001,甲企业
采购系统B,B-001,甲企业
采购系统A,A-002,乙企业
采购系统B,B-002,乙企业
采购系统A,A-003,丙企业
采购系统B,B-003,丙企业
采购系统A,A-004,丁企业
采购系统B,B-004,丁企业
采购系统A,A-005,戊企业
"""

A1, B1 = 1, 2
A2, B2 = 3, 4
A3 = 5


@pytest.fixture()
def paths(tmp_path: Path) -> tuple[Path, Path]:
    csv_path = tmp_path / "suppliers.csv"
    csv_path.write_text(SEED, encoding="utf-8")
    return tmp_path / "review.db", csv_path


def make_client(db_path: Path, csv_path: Path, instance: str = "test") -> TestClient:
    return TestClient(
        create_app(db_path=db_path, seed_csv=csv_path, seed_limit=100, instance=instance)
    )


def add_eq(client: TestClient, record_id: int, qa: int, ua: str, qb: int, ub: str):
    return client.post(
        "/api/equations",
        json={
            "record_id": record_id,
            "qty_a": qa,
            "unit_a": ua,
            "qty_b": qb,
            "unit_b": ub,
        },
    )


def query(client: TestClient, record_id: int, unit: str, quantity: int) -> dict:
    response = client.post(
        "/api/conversion/query",
        json={"record_id": record_id, "unit": unit, "quantity": quantity},
    )
    assert response.status_code == 200, response.text
    return response.json()["query"]


def state(client: TestClient) -> dict:
    return client.get("/api/state").json()["state"]


def group_key_of(state_data: dict, record_id: int) -> str:
    for group in state_data["groups"]:
        if any(member["id"] == record_id for member in group["members"]):
            return group["key"]
    raise AssertionError(record_id)


def add_and_confirm_same(client: TestClient, a: int, b: int) -> None:
    preview = client.post(
        "/api/previews",
        json={"operation": "add", "record_a": a, "record_b": b, "relation": "same"},
    ).json()["preview"]
    assert preview["confirmable"], preview
    response = client.post(
        f"/api/previews/{preview['preview_id']}/confirm",
        json={"snapshot_digest": preview["snapshot_digest"]},
    )
    assert response.status_code == 200, response.text


def withdraw_decision(client: TestClient, decision_id: int) -> None:
    preview = client.post(
        "/api/previews",
        json={"operation": "withdraw", "decision_id": decision_id},
    ).json()["preview"]
    assert preview["confirmable"], preview
    response = client.post(
        f"/api/previews/{preview['preview_id']}/confirm",
        json={"snapshot_digest": preview["snapshot_digest"]},
    )
    assert response.status_code == 200, response.text


# ------------------------------------------------------------- pure graph logic


def test_find_conversion_reverse_and_multihop():
    equations = [
        {"id": 1, "record_id": A1, "unit_a": "件", "qty_a": 24, "unit_b": "箱", "qty_b": 2},
        {"id": 2, "record_id": A1, "unit_a": "盒", "qty_a": 3, "unit_b": "箱", "qty_b": 1},
    ]
    adjacency = conversion.build_adjacency(equations)

    # Reverse traversal: equation stored as 24件=2箱, query 箱 -> 件.
    path = conversion.find_conversion(adjacency, "箱", "件")
    assert path is not None
    assert path.rate == Fraction(12)
    assert [edge["id"] for edge in path.edges] == [1]

    # Multi-hop: 盒 -> 箱 -> 件, rate 1/3 * 12 = 4.
    path = conversion.find_conversion(adjacency, "盒", "件")
    assert path is not None
    assert path.rate == Fraction(4)
    assert len(path.edges) == 2

    # Same unit is a zero-hop identity conversion.
    same = conversion.find_conversion(adjacency, "件", "件")
    assert same.rate == Fraction(1) and same.edges == ()

    assert conversion.find_conversion(adjacency, "托", "件") is None


def test_find_rate_conflict_detects_contradicting_cycles():
    consistent = [
        {"id": 1, "record_id": A1, "unit_a": "箱", "qty_a": 2, "unit_b": "件", "qty_b": 24},
        {"id": 2, "record_id": A1, "unit_a": "盒", "qty_a": 3, "unit_b": "箱", "qty_b": 1},
        {"id": 3, "record_id": A1, "unit_a": "盒", "qty_a": 1, "unit_b": "件", "qty_b": 4},
    ]
    assert conversion.find_rate_conflict(consistent) is None

    # 3盒=1箱 and 2箱=24件 imply 1盒=4件; claiming 1盒=9件 closes a bad cycle.
    contradictory = consistent + [
        {"id": 4, "record_id": A1, "unit_a": "盒", "qty_a": 1, "unit_b": "件", "qty_b": 9},
    ]
    conflict = conversion.find_rate_conflict(contradictory)
    assert conflict is not None
    assert conflict["unit_a"] == "盒" and conflict["unit_b"] == "件"
    assert conflict["existing_rate_b_over_a"] == Fraction(4)
    assert conflict["candidate_rate_b_over_a"] == Fraction(9)
    assert int(conflict["candidate"]["id"]) == 4


def test_fraction_text_uses_simplest_form():
    assert conversion.fraction_text(Fraction(36)) == "36"
    assert conversion.fraction_text(Fraction(4, 3)) == "4/3"
    assert conversion.fraction_text(Fraction(10, 4)) == "5/2"


# ------------------------------------------------------------- validation / CRUD


def test_equation_validation(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        bad_cases = [
            {"record_id": A1, "qty_a": 0, "unit_a": "箱", "qty_b": 1, "unit_b": "件"},
            {"record_id": A1, "qty_a": 1, "unit_a": "箱", "qty_b": -2, "unit_b": "件"},
            {"record_id": A1, "qty_a": 1, "unit_a": "  ", "qty_b": 1, "unit_b": "件"},
            {"record_id": A1, "qty_a": 1, "unit_a": "箱", "qty_b": 1, "unit_b": "箱"},
        ]
        for body in bad_cases:
            response = client.post("/api/equations", json=body)
            assert response.status_code == 400, body

        response = add_eq(client, 999, 1, "箱", 12, "件")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "unknown_record"

        # Non-integer quantities are rejected by the request schema.
        response = client.post(
            "/api/equations",
            json={"record_id": A1, "qty_a": 1.5, "unit_a": "箱",
                  "qty_b": 1, "unit_b": "件"},
        )
        assert response.status_code == 422

        # Strict typing: no boolean or string coercion for integer fields.
        for body in (
            {"record_id": A1, "qty_a": True, "unit_a": "箱", "qty_b": 1, "unit_b": "件"},
            {"record_id": A1, "qty_a": "2", "unit_a": "箱", "qty_b": 1, "unit_b": "件"},
            {"record_id": str(A1), "qty_a": 1, "unit_a": "箱", "qty_b": 1, "unit_b": "件"},
        ):
            assert client.post("/api/equations", json=body).status_code == 422
        assert client.post(
            "/api/conversion/query",
            json={"record_id": A1, "unit": "箱", "quantity": True},
        ).status_code == 422


def test_add_equation_persists_and_survives_restart(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        response = add_eq(client, A1, 2, "箱", 24, "件")
        assert response.status_code == 200, response.text
        equation = response.json()["equation"]
        assert equation["text"] == "2箱=24件"
        assert equation["record"]["id"] == A1
        assert state(client)["stats"]["equations"] == 1

    with make_client(db_path, csv_path, instance="restart") as client:
        data = state(client)
        assert data["stats"]["equations"] == 1
        group = next(g for g in data["groups"] if g["equation_count"] == 1)
        assert group["units"] == ["件", "箱"]
        assert group["equations"][0]["text"] == "2箱=24件"

        # Persisted in SQLite.
        conn = db.connect(db_path)
        try:
            rows = conn.execute("SELECT unit_a, qty_a, unit_b, qty_b FROM unit_equations").fetchall()
            assert len(rows) == 1
            assert (rows[0]["unit_a"], rows[0]["qty_a"], rows[0]["unit_b"], rows[0]["qty_b"]) == (
                "箱", 2, "件", 24,
            )
        finally:
            conn.close()


def test_record_equation_limit_is_20(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        for index in range(20):
            response = add_eq(client, A1, 1, f"单位{index}", 1, "件")
            assert response.status_code == 200, (index, response.text)
        response = add_eq(client, A1, 1, "单位20", 1, "件")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "equation_limit_reached"
        assert state(client)["stats"]["equations"] == 20


def test_duplicate_equation_rejected(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        assert add_eq(client, A1, 2, "箱", 24, "件").status_code == 200
        duplicate = add_eq(client, A1, 2, "箱", 24, "件")
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "equation_exists"
        # Same rate with reversed orientation is also a duplicate.
        reversed_duplicate = add_eq(client, A1, 24, "件", 2, "箱")
        assert reversed_duplicate.status_code == 409
        assert reversed_duplicate.json()["error"]["code"] == "equation_exists"
        assert state(client)["stats"]["equations"] == 1


def test_in_group_ratio_conflict_rejected_and_state_unchanged(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        assert add_eq(client, A1, 2, "箱", 24, "件").status_code == 200
        before = state(client)

        # Direct contradiction: existing implies 1箱=12件, this claims 13.
        response = add_eq(client, A1, 1, "箱", 13, "件")
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "unit_ratio_conflict"
        conflict = error["conflict"]
        assert conflict["candidate"]["text"] == "1箱=13件"
        assert [item["text"] for item in conflict["evidence"]] == ["2箱=24件"]
        assert conflict["existing_rate"] == "12"
        assert conflict["candidate_rate"] == "13"

        after = state(client)
        assert after["equations"] == before["equations"]
        assert after["state_version"] == before["state_version"]

        # A non-contradicting equation on another unit is accepted.
        assert add_eq(client, A1, 3, "盒", 1, "箱").status_code == 200


def test_conflict_check_spans_all_records_in_the_group(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        add_and_confirm_same(client, A1, B1)
        assert add_eq(client, A1, 1, "箱", 12, "件").status_code == 200
        # Equation attached to another record of the SAME group is validated too.
        response = add_eq(client, B1, 1, "箱", 13, "件")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "unit_ratio_conflict"


def test_delete_equation(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        equation_id = add_eq(client, A1, 2, "箱", 24, "件").json()["equation"]["id"]
        assert add_eq(client, A1, 1, "箱", 13, "件").status_code == 409

        response = client.post(
            "/api/equations/delete", json={"equation_id": equation_id}
        )
        assert response.status_code == 200
        assert response.json()["deleted_id"] == equation_id
        assert state(client)["stats"]["equations"] == 0

        # After removal the previously contradicting equation is acceptable.
        assert add_eq(client, A1, 1, "箱", 13, "件").status_code == 200

        missing = client.post(
            "/api/equations/delete", json={"equation_id": equation_id}
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "unknown_equation"


# ------------------------------------------------------------- conversion query


def test_query_missing_path_and_group_scoping(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        # Nothing at all yet.
        result = query(client, A1, "箱", 1)
        assert result["reachable"] is False
        assert result["reachable_units"] == []
        assert "件" in result["message"]

        add_eq(client, B1, 1, "托", 48, "件")  # different group from A1
        result = query(client, A1, "托", 1)
        assert result["reachable"] is False
        assert result["reachable_units"] == []

        # Partial component: 托 and 盒 are connected, but neither reaches 件.
        add_eq(client, A1, 1, "托", 2, "盒")
        result = query(client, A1, "托", 1)
        assert result["reachable"] is False
        assert result["reachable_units"] == ["件", "托", "盒"] or set(
            result["reachable_units"]
        ) == {"托", "盒"}
        assert "件" in result["message"]


def test_query_direct_reverse_multihop_and_exact_fraction(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        # Stored "backwards" to prove reverse traversal.
        add_eq(client, A1, 24, "件", 2, "箱")
        add_eq(client, A1, 3, "盒", 1, "箱")

        direct = query(client, A1, "箱", 3)
        assert direct["reachable"] is True
        assert direct["result_text"] == "36"
        assert direct["result_numerator"] == 36
        assert direct["result_denominator"] == 1
        assert direct["is_integer"] is True
        assert direct["hops"] == 1
        assert direct["used_equations"][0]["record"]["id"] == A1

        multi = query(client, A1, "盒", 2)
        assert multi["result_text"] == "8"  # 2盒 = 2/3箱 = 8件
        assert multi["hops"] == 2
        assert [eq["id"] for eq in multi["used_equations"]] == [2, 1]

        # Exact non-integer simplest fraction: add 3包=2件, ask for 2包 -> 4/3.
        add_eq(client, A1, 3, "包", 2, "件")
        fractional = query(client, A1, "包", 2)
        assert fractional["result_text"] == "4/3"
        assert (fractional["result_numerator"], fractional["result_denominator"]) == (4, 3)
        assert fractional["is_integer"] is False

        # Identity query for the target unit.
        identity = query(client, A1, "件", 7)
        assert identity["result_text"] == "7" and identity["hops"] == 0

        # Quantity must be a positive integer.
        bad = client.post(
            "/api/conversion/query",
            json={"record_id": A1, "unit": "箱", "quantity": 0},
        )
        assert bad.status_code == 400


def test_query_uses_only_current_group_sources_and_split(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        add_eq(client, A1, 1, "箱", 12, "件")
        add_eq(client, B1, 1, "托", 48, "件")

        # Separate groups: each query only sees its own group's equations.
        assert query(client, A1, "箱", 1)["reachable"] is True
        assert query(client, A1, "托", 1)["reachable"] is False
        assert query(client, B1, "托", 1)["reachable"] is True
        assert query(client, B1, "箱", 1)["reachable"] is False

        # Merge the groups: both equation sources become available to both.
        add_and_confirm_same(client, A1, B1)
        merged = query(client, A1, "托", 2)
        assert merged["reachable"] is True
        assert merged["result_text"] == "96"
        assert merged["used_equations"][0]["record"]["id"] == B1
        assert query(client, B1, "箱", 1)["result_text"] == "12"

        # Split again by withdrawing the only "same" decision: own sources only.
        decision_id = state(client)["decisions"][0]["id"]
        withdraw_decision(client, decision_id)
        assert query(client, A1, "托", 1)["reachable"] is False
        assert query(client, B1, "箱", 1)["reachable"] is False
        assert query(client, A1, "箱", 1)["result_text"] == "12"
        assert query(client, B1, "托", 1)["result_text"] == "48"


def test_query_result_carries_committed_state_version(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        add_eq(client, A1, 2, "箱", 24, "件")
        version = state(client)["state_version"]
        result = query(client, A1, "箱", 1)
        assert result["state_version"] == version


# ------------------------------------------------------------- merge integration


def test_merge_preview_blocked_by_unit_ratio_conflict(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        add_eq(client, A1, 1, "箱", 12, "件")
        add_eq(client, B1, 1, "箱", 13, "件")

        preview = client.post(
            "/api/previews",
            json={"operation": "add", "record_a": A1, "record_b": B1, "relation": "same"},
        ).json()["preview"]
        assert preview["confirmable"] is False
        assert preview["preview_id"] is None
        conflict = preview["conflict"]["equation_conflict"]
        assert conflict["code"] == "unit_ratio_conflict"
        assert conflict["candidate"]["text"] == "1箱=13件"
        assert [item["text"] for item in conflict["evidence"]] == ["1箱=12件"]

        # Nothing was persisted: groups stay split and both equations remain.
        data = state(client)
        assert group_key_of(data, A1) != group_key_of(data, B1)
        assert data["stats"]["same_decisions"] == 0
        assert data["stats"]["equations"] == 2


def test_merge_allowed_when_unit_sets_do_not_clash(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        add_eq(client, A1, 1, "箱", 12, "件")
        add_eq(client, B1, 1, "托", 48, "件")
        add_and_confirm_same(client, A1, B1)
        data = state(client)
        assert group_key_of(data, A1) == group_key_of(data, B1)
        group = next(g for g in data["groups"] if any(member["id"] == A1 for member in g["members"]))
        assert group["equation_count"] == 2


def test_equation_add_invalidates_related_merge_preview(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path, "one") as client_one, make_client(
        db_path, csv_path, "two"
    ) as client_two:
        add_eq(client_one, A1, 1, "箱", 12, "件")
        add_eq(client_two, B1, 1, "托", 48, "件")
        preview = client_one.post(
            "/api/previews",
            json={"operation": "add", "record_a": A1, "record_b": B1, "relation": "same"},
        ).json()["preview"]
        assert preview["confirmable"] is True

        # An equation landing in one of the involved groups changes the snapshot.
        add_eq(client_two, B1, 1, "包", 5, "件")
        response = client_one.post(
            f"/api/previews/{preview['preview_id']}/confirm",
            json={"snapshot_digest": preview["snapshot_digest"]},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "preview_stale"
        assert state(client_one)["stats"]["same_decisions"] == 0

        # An equation on an unrelated group must not invalidate the preview.
        preview2 = client_one.post(
            "/api/previews",
            json={"operation": "add", "record_a": A1, "record_b": B1, "relation": "same"},
        ).json()["preview"]
        add_eq(client_two, A3, 1, "包", 7, "件")
        response = client_one.post(
            f"/api/previews/{preview2['preview_id']}/confirm",
            json={"snapshot_digest": preview2["snapshot_digest"]},
        )
        assert response.status_code == 200, response.text


def test_confirm_revalidates_against_latest_equations(paths):
    """The core concurrency guarantee: preview and equation add race, and the
    merge confirm re-checks the prospective group inside its write transaction.
    """
    db_path, csv_path = paths
    with make_client(db_path, csv_path, "one") as client_one, make_client(
        db_path, csv_path, "two"
    ) as client_two:
        add_eq(client_one, A1, 1, "箱", 12, "件")
        add_eq(client_two, B1, 1, "托", 48, "件")
        preview = client_one.post(
            "/api/previews",
            json={"operation": "add", "record_a": A1, "record_b": B1, "relation": "same"},
        ).json()["preview"]

        # After the preview, another process makes the prospective group clash.
        add_eq(client_two, B1, 1, "箱", 13, "件")
        response = client_one.post(
            f"/api/previews/{preview['preview_id']}/confirm",
            json={"snapshot_digest": preview["snapshot_digest"]},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "merge_unit_ratio_conflict"

        data = state(client_one)
        assert group_key_of(data, A1) != group_key_of(data, B1)
        assert data["stats"]["same_decisions"] == 0
        assert data["stats"]["equations"] == 3

        # A fresh preview now reports the conflict directly and cannot confirm.
        stale_preview = client_two.post(
            "/api/previews",
            json={"operation": "add", "record_a": A1, "record_b": B1, "relation": "same"},
        ).json()["preview"]
        assert stale_preview["confirmable"] is False


def test_cross_process_conflicting_equation_add_is_serialized(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path, "one") as client_one, make_client(
        db_path, csv_path, "two"
    ) as client_two:
        assert add_eq(client_one, A1, 1, "箱", 12, "件").status_code == 200
        # BEGIN IMMEDIATE serializes the writers; the second validation sees
        # the committed equation and rejects the contradicting rate.
        response = add_eq(client_two, A1, 1, "箱", 13, "件")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "unit_ratio_conflict"
        assert state(client_one)["stats"]["equations"] == 1


def test_failed_writes_leave_database_unchanged(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        add_eq(client, A1, 2, "箱", 24, "件")
        before = state(client)
        for body in (
            {"record_id": A1, "qty_a": 1, "unit_a": "箱", "qty_b": 13, "unit_b": "件"},
            {"record_id": 999, "qty_a": 1, "unit_a": "箱", "qty_b": 1, "unit_b": "件"},
        ):
            assert client.post("/api/equations", json=body).status_code in (404, 409)
        after = state(client)
        assert after["equations"] == before["equations"]
        assert after["state_version"] == before["state_version"]

        conn = db.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) AS c FROM unit_equations").fetchone()["c"]
        finally:
            conn.close()
        assert count == 1


# ------------------------------------------------------------- pages


def test_pages_and_navigation(paths):
    db_path, csv_path = paths
    with make_client(db_path, csv_path) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert 'href="/convert"' in home.text
        assert "/static/app.css" in home.text

        convert_page = client.get("/convert")
        assert convert_page.status_code == 200
        assert "包装单位换算台" in convert_page.text
        assert "2箱=24件" in convert_page.text

        css = client.get("/static/app.css")
        assert css.status_code == 200
        assert "text/css" in css.headers["content-type"]
