"""年度活动数据批量核验：批量核验 + 重算 + 报告/配额/仪表盘联动，失败整体回滚。

覆盖：
- 按 ID 列表 / 按企业+年度两种批量核验模式，核验后同事务重算；
- 报告草稿联动：draft 刷新快照、submitted 退回草稿、approved 不触碰；
- 批准前未核验拦截与配额冻结的全链路打通；
- 批量失败（缺失 ID、跨企业年度、重算异常）整体回滚，不留半成品状态；
- 仪表盘统计（待核验条数、排放量、冻结额）随批量操作一致更新；
- 并发批量核验只生效一次，最终状态一致。
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models import (
    ActivityData,
    AllowanceAccount,
    ComplianceRecord,
    EmissionResult,
    MrvReport,
)
from app.services.calculation_service import annual_total, recalc_company_year
from app.services.mrv_service import approve_report, generate_report, submit_report
from app.services.quota_service import allocate_quota
from app.services.stats_service import dashboard_stats
from app.services.verification_service import batch_verify_activities


def approx(value, rel=1e-6):
    return pytest.approx(float(value), rel=rel)


def _add_activity(db, seed, scope, year, atype, qty, unit="t", verified=0):
    act = ActivityData(
        company_id=seed["company"].id,
        scope_id=scope.id,
        year=year,
        period="monthly",
        activity_type=atype,
        unit=unit,
        quantity=qty,
        data_source="测试台账",
        verified=verified,
    )
    db.add(act)
    db.commit()
    return act


class TestBatchVerify:
    def test_batch_verify_by_ids_recalculates(self, db, seed):
        """按 ID 批量核验：全部标记已核验，同事务重算并计入核算结果。"""
        _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh", verified=1)
        a1 = _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 3000, "MWh")
        a2 = _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 1000, "MWh")
        recalc_company_year(db, seed["company"].id, 2025)
        assert annual_total(db, seed["company"].id, 2025) == approx(2000 * 0.5703)

        result = batch_verify_activities(db, activity_ids=[a1.id, a2.id])

        assert result["verified_count"] == 2
        assert result["already_verified"] == 0
        assert result["recalculated"] is True
        assert result["result_count"] == 3
        assert result["unverified_remaining"] == 0
        db.refresh(a1)
        db.refresh(a2)
        assert a1.verified == 1 and a2.verified == 1
        assert annual_total(db, seed["company"].id, 2025) == approx(6000 * 0.5703)

    def test_batch_verify_by_company_year_filter(self, db, seed):
        """按企业+年度核验全部待核验记录。"""
        for qty in (1000, 2000, 3000):
            _add_activity(db, seed, seed["scope2"], 2025, "外购电力", qty, "MWh")
        _add_activity(db, seed, seed["scope2"], 2024, "外购电力", 9999, "MWh")  # 其他年度不受影响

        result = batch_verify_activities(db, company_id=seed["company"].id, year=2025)

        assert result["verified_count"] == 3
        assert result["result_count"] == 3
        assert annual_total(db, seed["company"].id, 2025) == approx(6000 * 0.5703)
        assert annual_total(db, seed["company"].id, 2024) == 0  # 2024 年未核验，不进入核算

    def test_batch_verify_rolls_back_when_id_missing(self, db, seed):
        """批次含不存在的 ID：整批拒绝，核验标记与核算结果均不变。"""
        a1 = _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh")

        with pytest.raises(ValueError, match="不存在"):
            batch_verify_activities(db, activity_ids=[a1.id, 99999])

        db.refresh(a1)
        assert a1.verified == 0
        assert db.query(EmissionResult).count() == 0

    def test_batch_verify_rejects_cross_year(self, db, seed):
        """跨年度批次整批拒绝并回滚。"""
        a1 = _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh")
        a2 = _add_activity(db, seed, seed["scope2"], 2024, "外购电力", 3000, "MWh")

        with pytest.raises(ValueError, match="同一企业同一年度"):
            batch_verify_activities(db, activity_ids=[a1.id, a2.id])

        db.refresh(a1)
        db.refresh(a2)
        assert a1.verified == 0 and a2.verified == 0

    def test_batch_verify_idempotent_when_already_verified(self, db, seed):
        """全部已核验：幂等空操作，不触发重算。"""
        a1 = _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh", verified=1)
        recalc_company_year(db, seed["company"].id, 2025)

        result = batch_verify_activities(db, activity_ids=[a1.id])

        assert result["verified_count"] == 0
        assert result["already_verified"] == 1
        assert result["recalculated"] is False
        assert db.query(EmissionResult).count() == 1

    def test_batch_verify_empty_filter_rejected(self, db, seed):
        """该年度无待核验数据时明确拒绝。"""
        with pytest.raises(ValueError, match="没有待核验"):
            batch_verify_activities(db, company_id=seed["company"].id, year=2025)

    def test_batch_verify_refreshes_draft_report(self, db, seed):
        """草稿报告随批量核验刷新排放快照，保持草稿状态。"""
        _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh", verified=1)
        pending = _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 3000, "MWh")
        recalc_company_year(db, seed["company"].id, 2025)
        report = generate_report(db, seed["company"].id, 2025)
        assert float(report.total_emission) == approx(2000 * 0.5703)

        result = batch_verify_activities(db, activity_ids=[pending.id])

        db.refresh(report)
        assert report.status == "draft"
        assert float(report.total_emission) == approx(5000 * 0.5703)
        assert result["report"]["status"] == "draft"
        assert result["report"]["total_emission"] == approx(5000 * 0.5703)
        assert result["report_reset"] is False

    def test_batch_verify_resets_submitted_report_to_draft(self, db, seed):
        """已提交报告：底层数据变化后退回草稿并刷新快照，需重新提交核查。"""
        _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh", verified=1)
        pending = _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 3000, "MWh")
        recalc_company_year(db, seed["company"].id, 2025)
        report = generate_report(db, seed["company"].id, 2025)
        submit_report(db, report)

        result = batch_verify_activities(db, activity_ids=[pending.id])

        db.refresh(report)
        assert report.status == "draft"
        assert float(report.total_emission) == approx(5000 * 0.5703)
        assert result["report_reset"] is True
        assert "退回草稿" in result["warning"]

    def test_batch_verify_keeps_approved_report_untouched(self, db, seed):
        """已批准报告不触碰：配额冻结与履约结果不变，响应附警告。"""
        _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh", verified=1)
        recalc_company_year(db, seed["company"].id, 2025)
        allocate_quota(db, seed["company"].id, 2025, baseline=1000, allocation_amount=5000)
        report = generate_report(db, seed["company"].id, 2025)
        submit_report(db, report)
        approve_report(db, report, verifier_id=1)
        record = db.query(ComplianceRecord).filter(ComplianceRecord.is_active == 1).one()
        frozen_before = float(record.frozen_amount)

        pending = _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 3000, "MWh")
        result = batch_verify_activities(db, activity_ids=[pending.id])

        db.refresh(report)
        db.refresh(record)
        assert report.status == "approved"
        assert float(report.total_emission) == approx(2000 * 0.5703)  # 快照不变
        assert float(record.frozen_amount) == frozen_before
        assert result["report"]["status"] == "approved"
        assert "已批准" in result["warning"]

    def test_batch_verify_unblocks_approval_and_freeze(self, db, seed):
        """全链路：未核验拦截批准 → 批量核验 → 重算 → 重新生成 → 批准冻结 → 仪表盘一致。"""
        _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh", verified=1)
        _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 3000, "MWh")
        recalc_company_year(db, seed["company"].id, 2025)
        allocate_quota(db, seed["company"].id, 2025, baseline=1000, allocation_amount=10000)
        report = generate_report(db, seed["company"].id, 2025)
        submit_report(db, report)

        # 批准前拦截：存在未核验数据
        with pytest.raises(ValueError, match="未核验"):
            approve_report(db, report, verifier_id=1)

        # 批量核验 + 重算：已提交报告退回草稿并刷新快照
        result = batch_verify_activities(db, company_id=seed["company"].id, year=2025)
        assert result["unverified_remaining"] == 0
        db.refresh(report)
        assert report.status == "draft"
        assert float(report.total_emission) == approx(5000 * 0.5703)

        # 重新提交并批准：配额按新快照冻结
        submit_report(db, report)
        approve_report(db, report, verifier_id=1)
        assert report.status == "approved"
        record = db.query(ComplianceRecord).filter(ComplianceRecord.is_active == 1).one()
        assert float(record.verified_emission) == approx(5000 * 0.5703)
        assert float(record.frozen_amount) == approx(5000 * 0.5703)

        # 仪表盘统计联动：待核验清零、排放量与冻结额同步
        stats = dashboard_stats(db, 2025)
        assert stats["unverified_activity"] == 0
        assert stats["emission_total"] == approx(5000 * 0.5703)
        assert stats["frozen_total"] == approx(5000 * 0.5703)

    def test_batch_verify_rollback_on_recalc_failure(self, db, seed, monkeypatch):
        """重算环节异常：核验标记、核算结果、报告快照全部回滚。"""
        _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh", verified=1)
        pending = _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 3000, "MWh")
        recalc_company_year(db, seed["company"].id, 2025)
        report = generate_report(db, seed["company"].id, 2025)
        submit_report(db, report)

        import app.services.verification_service as vs

        def boom(_db, _cid, _yr):
            raise RuntimeError("模拟重算失败")

        monkeypatch.setattr(vs, "recalc_company_year_tx", boom)
        with pytest.raises(RuntimeError, match="模拟重算失败"):
            batch_verify_activities(db, activity_ids=[pending.id])

        db.expire_all()
        assert db.get(ActivityData, pending.id).verified == 0
        # 旧核算结果未被清除（先清后算在同一事务内回滚）
        assert db.query(EmissionResult).count() == 1
        assert annual_total(db, seed["company"].id, 2025) == approx(2000 * 0.5703)
        # 报告保持已提交，快照未变
        db.refresh(report)
        assert report.status == "submitted"
        assert float(report.total_emission) == approx(2000 * 0.5703)

    def test_dashboard_unverified_count_tracks_batch(self, db, seed):
        """仪表盘待核验统计随批量核验同步清零。"""
        _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 2000, "MWh")
        _add_activity(db, seed, seed["scope2"], 2025, "外购电力", 3000, "MWh")
        assert dashboard_stats(db, 2025)["unverified_activity"] == 2

        batch_verify_activities(db, company_id=seed["company"].id, year=2025)

        stats = dashboard_stats(db, 2025)
        assert stats["unverified_activity"] == 0
        assert stats["total_results"] == 2
        assert stats["emission_total"] == approx(5000 * 0.5703)


class TestBatchVerifyConcurrency:
    """并发批量核验：进程内企业年度键串行化，重复批次幂等，最终状态一致。"""

    @pytest.fixture()
    def file_session(self, tmp_path):
        engine = create_engine(
            f"sqlite:///{tmp_path / 'batch_verify.db'}",
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(engine, "connect")
        def _busy_timeout(dbapi_conn, _rec):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine)
        yield factory
        engine.dispose()

    def test_concurrent_batch_verify_only_one_takes_effect(self, file_session):
        from app.models import CalculationMethod, Company, EmissionFactor, EmissionScope

        db = file_session()
        company = Company(code="C-CC", name="并发企业", industry="电力", region="测试")
        db.add(company)
        db.flush()
        scope = EmissionScope(company_id=company.id, scope="2", category="外购电力", name="厂区用电")
        db.add(scope)
        db.flush()
        db.add(
            EmissionFactor(
                factor_code="ELEC-CC", name="外购电力", scope="2", unit="tCO2/MWh",
                value=0.5703, source="电网因子", valid_from="2024-01-01", valid_to="2025-12-31",
            )
        )
        db.add(CalculationMethod(method_code="ELEC-CC-M", name="因子法", scope="2", formula_type="activity_factor"))
        for i in range(6):
            db.add(
                ActivityData(
                    company_id=company.id, scope_id=scope.id, year=2025, period="monthly",
                    activity_type="外购电力", unit="MWh", quantity=1000 + i,
                    data_source="并发测试", verified=0,
                )
            )
        db.commit()
        company_id = company.id
        db.close()

        outcomes = []

        def worker():
            session = file_session()
            try:
                result = batch_verify_activities(session, company_id=company_id, year=2025)
                return ("ok", result["verified_count"])
            except ValueError as e:
                return ("rejected", str(e))
            finally:
                session.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: worker(), range(2)))

        # 恰有一个批次核验 6 条；另一批次要么空操作（核验 0 条），要么被明确拒绝
        verified_counts = sorted(c for status, c in outcomes if status == "ok")
        assert verified_counts and verified_counts[-1] == 6
        assert sum(verified_counts) == 6

        check = file_session()
        try:
            acts = check.query(ActivityData).filter(ActivityData.company_id == company_id).all()
            assert all(a.verified == 1 for a in acts)
            results = check.query(EmissionResult).filter(EmissionResult.company_id == company_id).all()
            assert len(results) == 6  # 重算幂等：先清后算，无重复结果
            expected = sum(1000 + i for i in range(6)) * 0.5703
            assert annual_total(check, company_id, 2025) == approx(expected)
            assert dashboard_stats(check, 2025)["unverified_activity"] == 0
        finally:
            check.close()
