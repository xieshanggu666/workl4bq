"""年度批量核验 API 集成测试：鉴权边界、响应结构、失败回滚与仪表盘联动。"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.security import hash_password
from app.main import app
from app.models import (
    ActivityData,
    CalculationMethod,
    Company,
    EmissionFactor,
    EmissionScope,
    User,
)


@pytest.fixture()
def ctx():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False)
    Base.metadata.create_all(engine)

    db = TestingSession()
    pwd_hash, salt = hash_password("123456")

    company = Company(code="B-001", name="批量核验企业", industry="电力", region="华东")
    db.add(company)
    db.flush()
    scope = EmissionScope(company_id=company.id, scope="2", category="外购电力", name="厂区用电")
    db.add(scope)
    db.flush()
    db.add(
        EmissionFactor(
            factor_code="ELEC-BATCH", name="外购电力", scope="2", unit="tCO2/MWh",
            value=0.5703, source="电网因子", valid_from="2024-01-01", valid_to="2025-12-31",
        )
    )
    db.add(CalculationMethod(method_code="ELEC-BATCH-M", name="因子法", scope="2", formula_type="activity_factor"))

    users = [
        User(username="ent1", display_name="企业", role="enterprise", company_id=company.id, password_hash=pwd_hash, salt=salt),
        User(username="admin", display_name="监管员", role="admin", password_hash=pwd_hash, salt=salt),
        User(username="verifier", display_name="核查员", role="verifier", password_hash=pwd_hash, salt=salt),
    ]
    db.add_all(users)
    db.commit()
    ids = {"company": company.id, "scope": scope.id}
    db.close()

    def override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client, ids, TestingSession
    app.dependency_overrides.clear()
    engine.dispose()


def login(client, username):
    client.post("/api/auth/login", json={"username": username, "password": "123456"})


def _add_activity(client, ids, qty, year=2025):
    """以企业身份录入一条待核验活动数据，返回记录 ID。"""
    login(client, "ent1")
    res = client.post("/api/activity", json={
        "scope_id": ids["scope"], "year": year, "period": "monthly",
        "activity_type": "外购电力", "unit": "MWh", "quantity": qty,
        "data_source": "关口计量表",
    })
    assert res.status_code == 200, res.text
    return res.json()["id"]


def test_api_batch_verify_flow_and_response(ctx):
    """批量核验 → 重算 → 台账/仪表盘联动，响应结构完整。"""
    client, ids, TestingSession = ctx
    for qty in (1000, 2000, 3000):
        _add_activity(client, ids, qty)

    login(client, "verifier")
    res = client.post("/api/activity/verify-batch", json={"company_id": ids["company"], "year": 2025})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["verified_count"] == 3
    assert data["recalculated"] is True
    assert data["result_count"] == 3
    assert data["unverified_remaining"] == 0
    assert data["company_id"] == ids["company"]
    assert data["year"] == 2025

    # 台账全部已核验
    res = client.get(f"/api/activity?company_id={ids['company']}&year=2025")
    assert all(a["verified"] == 1 for a in res.json())

    # 核算结果与仪表盘统计联动
    res = client.get(f"/api/companies/{ids['company']}/results?year=2025")
    assert len(res.json()) == 3
    stats = client.get("/api/dashboard/stats?year=2025").json()
    assert stats["unverified_activity"] == 0
    assert stats["total_results"] == 3
    assert stats["emission_total"] == pytest.approx(6000 * 0.5703, rel=1e-6)


def test_api_batch_verify_by_ids(ctx):
    """按 ID 列表批量核验。"""
    client, ids, _ = ctx
    a1 = _add_activity(client, ids, 1000)
    a2 = _add_activity(client, ids, 2000)

    login(client, "verifier")
    res = client.post("/api/activity/verify-batch", json={"ids": [a1, a2]})
    assert res.status_code == 200, res.text
    assert res.json()["verified_count"] == 2


def test_api_batch_verify_enterprise_forbidden(ctx):
    """企业角色无权批量核验。"""
    client, ids, _ = ctx
    _add_activity(client, ids, 1000)
    login(client, "ent1")
    res = client.post("/api/activity/verify-batch", json={"company_id": ids["company"], "year": 2025})
    assert res.status_code == 403


def test_api_batch_verify_invalid_ids_rollback(ctx):
    """批次含不存在 ID：400 且整批回滚，已有记录保持未核验。"""
    client, ids, TestingSession = ctx
    a1 = _add_activity(client, ids, 1000)

    login(client, "verifier")
    res = client.post("/api/activity/verify-batch", json={"ids": [a1, 99999]})
    assert res.status_code == 400
    assert "不存在" in res.json()["detail"]

    db = TestingSession()
    try:
        act = db.get(ActivityData, a1)
        assert act.verified == 0
    finally:
        db.close()


def test_api_batch_verify_empty_rejected(ctx):
    """无待核验数据 / 缺参数：400。"""
    client, ids, _ = ctx
    login(client, "verifier")
    res = client.post("/api/activity/verify-batch", json={"company_id": ids["company"], "year": 2025})
    assert res.status_code == 400
    assert "没有待核验" in res.json()["detail"]

    res = client.post("/api/activity/verify-batch", json={})
    assert res.status_code == 400


def test_api_batch_verify_unauthenticated(ctx):
    """未登录拒绝。"""
    client, ids, _ = ctx
    res = client.post("/api/activity/verify-batch", json={"company_id": ids["company"], "year": 2025})
    assert res.status_code == 401
