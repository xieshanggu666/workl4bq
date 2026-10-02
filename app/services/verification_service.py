"""活动数据批量核验：年度批量核验 + 重算 + 报告草稿联动，单事务失败整体回滚。

将原先逐条 ``POST /api/activity/{id}/verify`` 的核验方式升级为年度批量操作：
一次调用完成"核验标记 → 排放重算 → MRV 报告草稿刷新"的完整链路，
并与批准前未核验拦截、配额冻结、仪表盘统计保持同一数据源：

- 核验与重算在同一事务提交，任一步失败（含并发冲突、下游异常）整体回滚，
  绝不出现"部分记录已核验、核算结果未更新"的半成品状态；
- 重算后联动刷新该年度未定型报告（draft 刷新快照、submitted 退回草稿），
  维持"提交/批准的报告快照 = 当前核算结果"的不变量；
- 已批准报告不触碰：其排放快照已转化为配额冻结与履约结果，须先冲正；
- 通过 ``clear:<company>:<year>`` 键与报告批准/履约清缴串行化，
  并发批量核验只生效一次，重算幂等（先清后算）。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.ledger import company_clear_key, locked_accounts, transactional
from app.models.emission import ActivityData
from app.models.report import MrvReport
from app.services.calculation_service import (
    count_unverified_activities,
    recalc_company_year_tx,
)
from app.services.mrv_service import refresh_report_draft_tx


def _resolve_targets(
    db: Session,
    activity_ids: list[int] | None,
    company_id: int | None,
    year: int | None,
) -> list[ActivityData]:
    """解析批量核验目标集合（只读校验，失败即拒绝，不产生任何写入）。"""
    if activity_ids:
        unique_ids = sorted(set(activity_ids))
        activities = (
            db.query(ActivityData)
            .filter(ActivityData.id.in_(unique_ids))
            .order_by(ActivityData.id.asc())
            .all()
        )
        found = {a.id for a in activities}
        missing = [i for i in unique_ids if i not in found]
        if missing:
            raise ValueError(f"活动数据不存在：{missing}，批量核验已取消，未做任何变更")
        return activities

    if company_id is None or year is None:
        raise ValueError("请指定企业ID与年度，或直接指定活动数据ID列表")
    activities = (
        db.query(ActivityData)
        .filter(
            ActivityData.company_id == company_id,
            ActivityData.year == year,
            ActivityData.verified == 0,
        )
        .order_by(ActivityData.id.asc())
        .all()
    )
    if not activities:
        raise ValueError(f"该企业{year}年度没有待核验的活动数据")
    return activities


def batch_verify_activities(
    db: Session,
    *,
    activity_ids: list[int] | None = None,
    company_id: int | None = None,
    year: int | None = None,
    recalculate: bool = True,
) -> dict:
    """批量核验活动数据并（默认）同事务重算、联动刷新 MRV 报告草稿。

    - ``activity_ids``：直接指定记录批量核验；所有记录必须属于同一企业同一年度，
      否则整批拒绝（保证重算与报告联动的边界唯一）；
    - 缺省按 ``company_id + year`` 核验该年度全部待核验记录；
    - 已核验记录幂等跳过；目标全部已核验时为空操作（不触发重算）；
    - ``recalculate=False`` 时仅打核验标记，响应附待重算提示。

    返回汇总字典（核验条数、重算结果条数、剩余未核验、报告状态与警告）。
    """
    activities = _resolve_targets(db, activity_ids, company_id, year)

    pairs = {(a.company_id, a.year) for a in activities}
    if len(pairs) > 1:
        raise ValueError("批量核验仅支持同一企业同一年度的活动数据，请分批操作")
    cid, yr = next(iter(pairs))
    target_ids = [a.id for a in activities]

    verified_count = 0
    recalculated = False
    result_count = 0
    report_reset = False

    # 与报告批准/履约清缴共用企业年度键：批量核验-重算-报告刷新与批准冻结互斥，
    # 并发批次串行执行，锁内重查保证重复批次幂等空操作。
    with locked_accounts([company_clear_key(cid, yr)]):
        db.expire_all()
        pending = (
            db.query(ActivityData)
            .filter(ActivityData.id.in_(target_ids), ActivityData.verified == 0)
            .all()
        )
        if pending:
            with transactional(db):
                for act in pending:
                    act.verified = 1
                verified_count = len(pending)
                if recalculate:
                    result_count = recalc_company_year_tx(db, cid, yr)
                    _, report_reset = refresh_report_draft_tx(db, cid, yr)
                    recalculated = True

    unverified_remaining = count_unverified_activities(db, cid, yr)
    report = (
        db.query(MrvReport)
        .filter(MrvReport.company_id == cid, MrvReport.year == yr)
        .first()
    )

    warning = ""
    if report is not None and report.status == "approved":
        warning = "该年度报告已批准：本次核验/重算不影响已冻结配额与履约结果，如需调整请先冲正报告"
    elif report_reset:
        warning = "活动数据变化：原已提交报告已退回草稿并刷新快照，需重新提交核查"
    elif verified_count and not recalculated:
        warning = "本次未重算：新核验数据尚未计入核算结果，请触发核算并重新生成报告"

    return {
        "company_id": cid,
        "year": yr,
        "verified_count": verified_count,
        "already_verified": len(target_ids) - verified_count,
        "recalculated": recalculated,
        "result_count": result_count,
        "unverified_remaining": unverified_remaining,
        "report": None
        if report is None
        else {
            "id": report.id,
            "status": report.status,
            "total_emission": float(report.total_emission),
        },
        "report_reset": report_reset,
        "warning": warning,
    }
