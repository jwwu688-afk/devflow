"""评测（Eval）接口，挂载前缀 /api/evals。

这里是「AI 回答质量」的回归测试台，用来回答一个问题：改了检索/提示词/模型之后，
效果是变好了还是变差了？三类能力：
    1. 跑评测：基础评测（同步）与 RAGAS 质量评测（后台异步 + 轮询进度）；
    2. 对比评测：拿两次结果算指标差值和逐题通过情况，并给出「能否发布」的门禁结论；
    3. 落地问题：把一次失败的评测一键转成 Issue 草稿（走安全草稿流程，不直接建 issue）。

评测结果是整块 JSON 存在 EvalRun.result_json 里，而不是拆成结构化表——
因为不同评测器产出的字段差异很大，拆表反而要频繁改 schema。

异步任务的关键约定：后台任务用 SessionLocal 自建 session（请求的 db 在响应后就关了），
并且每个阶段都把进度写回 result_json，前端靠轮询 GET /{eval_id} 展示进度。
"""

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models import ActionDraft, EvalRun, Repository, User
from app.db.session import SessionLocal, get_db
from app.schemas.evals import EvalIssueDraftRequest, EvalRunRequest, EvalRunResponse
from app.services.evals.ragas_runner import (
    REQUIRED_RAGAS_METRICS,
    RagEvalCase,
    rescore_rag_result,
    run_rag_quality_eval,
)
from app.services.evals.suites import AUTO_SMOKE_SUITE_ID, list_eval_suites, load_eval_suite
from app.services.evals.runner import run_basic_eval
from app.services.permissions import require_permission, write_audit_log

router = APIRouter()
# 后台任务里的异常无法再抛给客户端（响应早就返回了），只能靠日志留痕。
logger = logging.getLogger(__name__)


def _request_case(item) -> RagEvalCase:
    """把请求里的用例（pydantic 模型）转成评测器内部的 RagEvalCase。

    逐个字段显式搬运，不做 **item.dict()：请求模型可能比内部模型多字段
    （比如前端传了展示用的 name），展开会直接报 TypeError。
    """
    return RagEvalCase(
        id=item.id,
        question=item.question,
        reference=item.reference,
        expected_source_ids=item.expected_source_ids,
        source_type=item.source_type,
        required_tools=item.required_tools,
        expected_tool_calls=item.expected_tool_calls,
        kind=item.kind,
        must_include=item.must_include,
        must_not_include=item.must_not_include,
        answer_regex=item.answer_regex,
        source_file=item.source_file,
    )


def _custom_evaluation_set(payload: EvalRunRequest) -> dict:
    """为「API 直接提交用例」的场景生成一份评测集元信息。

    返回：形如 {id, name, label, source, sha256, case_count} 的字典。
    说明：
        sha256 是对用例列表做「键排序 + 无多余空白」的规范序列化后求的哈希，
        目的是给评测集一个稳定的指纹：两次提交完全相同的用例会得到同一个 sha256，
        对比接口就靠它判断「是不是同一套题」（不同题的结果没有可比性）。
    """
    serialized = [item.model_dump(mode="json") for item in payload.cases or []]
    canonical = json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        # 三者都回退到同一个默认名：id 是内部标识、name 用于展示、label 用于中文标题。
        "id": payload.name or "custom-cases",
        "name": payload.name or "custom-cases",
        "label": payload.name or "自定义评测集",
        "description": "通过 API 提交的固定评测用例",
        "source": "api_cases",
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "case_count": len(serialized),
    }


@router.get("/suites")
async def get_eval_suites(
    repo_id: UUID = Query(...),
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> list[dict]:
    """GET /api/evals/suites?repo_id=...：列出该仓库可用的评测集。

    入参：repo_id 必填查询参数；db / 当前用户由依赖注入（用户仅用于权限校验）。
    返回：评测集元信息列表。
    异常：仓库不存在（service 抛 ValueError）→ 404。
    副作用：只读（会扫描评测集文件/配置）。
    """
    try:
        return list_eval_suites(db, repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/run", response_model=EvalRunResponse)
async def run_eval(
    payload: EvalRunRequest,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> EvalRunResponse:
    """POST /api/evals/run：跑一次「基础评测」（同步）。

    入参：payload.repo_id 可选（不传则跑不依赖仓库的通用检查）。
    返回：EvalRunResponse(eval_id, status="completed", result)。
    异常：底层 runner 抛出的异常直接向上（全局 500）。
    副作用：**同步执行**，请求会一直阻塞到评测跑完；结果写入 EvalRun。
    注意：这个接口返回的 status 永远是 completed——因为它是同步的，
        失败会直接抛异常而不是落到一个 failed 状态里。
    """
    result = await run_basic_eval(db, str(payload.repo_id) if payload.repo_id else None)
    eval_run = EvalRun(name=payload.name, result_json=result)
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    return EvalRunResponse(
        eval_id=str(eval_run.id),
        status="completed",
        result=result,
    )


@router.post("/rag/run", response_model=EvalRunResponse)
async def run_rag_eval(
    payload: EvalRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> EvalRunResponse:
    """POST /api/evals/rag/run：提交一次 RAGAS 质量评测（异步）。

    入参：
        payload.repo_id: 必填，评测要针对具体仓库的文档。
        payload.cases: 显式用例列表（与 suite_id 互斥）。
        payload.suite_id: 预置评测集 ID；都不传则用「自动冒烟」模式。
        payload.top_k / run_judge: 检索条数与是否启用 LLM Judge。
    返回：EvalRunResponse(status="queued")，真正的执行在后台。
    异常：
        没有 repo_id → 400；
        仓库不存在 → 404；
        cases 与 suite_id 同时给 → 400（语义冲突，无法判断以谁为准）；
        suite_id 不存在 → 400。
    副作用：写一条 status=queued 的 EvalRun，并把耗时的评测丢到 BackgroundTasks。
    三种模式的选择逻辑（见下方 if/elif/else）：
        cases     → 用请求里给的固定用例，评测集指纹由用例内容算出；
        suite_id  → 用预置评测集；但自动冒烟集要求 cases=None，让 runner 现出题；
        都不给    → 自动冒烟：基于当前仓库文档临时生成问题。
    """
    if payload.repo_id is None:
        raise HTTPException(status_code=400, detail="RAG Eval 需要 repo_id")
    if db.get(Repository, payload.repo_id) is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    if payload.cases is not None and payload.suite_id:
        raise HTTPException(status_code=400, detail="cases 与 suite_id 不能同时提交")
    if payload.cases is not None:
        cases = [_request_case(item) for item in payload.cases]
        evaluation_set = _custom_evaluation_set(payload)
    elif payload.suite_id:
        try:
            suite = load_eval_suite(db, payload.repo_id, payload.suite_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # 自动冒烟集的用例是运行时生成的，所以即使加载到了 suite 也要把 cases 置 None，
        # 交给 run_rag_quality_eval 自己出题；普通预置集则直接用它的固定用例。
        cases = None if payload.suite_id == AUTO_SMOKE_SUITE_ID else suite.cases
        evaluation_set = suite.metadata
    else:
        cases = None
        evaluation_set = {
            "id": AUTO_SMOKE_SUITE_ID,
            "name": AUTO_SMOKE_SUITE_ID,
            "label": "自动文档冒烟测试",
            "description": "未指定固定评测集，使用当前仓库文档临时出题。",
            "source": "auto_smoke",
            # 每次出题都不同，所以没有稳定指纹，case_count 也无法预知。
            "sha256": None,
            "case_count": None,
        }
    # 先写一个 queued 的占位结果，前端立刻就能按 eval_id 轮询到「排队中」。
    result = _job_result(
        status="queued",
        stage="queued",
        message="评测任务已提交，正在等待后台执行。",
        repo_id=str(payload.repo_id),
        evaluation_set=evaluation_set,
    )
    eval_run = EvalRun(name=payload.name or "rag_eval", result_json=result)
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    background_tasks.add_task(
        _execute_rag_eval_job,
        str(eval_run.id),
        str(payload.repo_id),
        cases,
        payload.top_k,
        payload.run_judge,
        evaluation_set,
    )
    return EvalRunResponse(eval_id=str(eval_run.id), status="queued", result=result)


async def _execute_rag_eval_job(
    eval_id: str,
    repo_id: str,
    cases: list[RagEvalCase] | None,
    limit: int,
    run_judge: bool,
    evaluation_set: dict,
) -> None:
    """后台执行一次完整的 RAGAS 评测，并把进度/结果写回 EvalRun。

    入参：
        eval_id: 目标 EvalRun 的字符串 ID（后台任务里跨 session 传递，故用 str）。
        cases: 固定用例；None 表示由 runner 现出题（自动冒烟模式）。
        limit: top_k。
        run_judge: 是否启用 LLM Judge 打分（关掉可省大量 token，但拿不到 RAGAS 指标）。
        evaluation_set: 评测集元信息，会一起写进结果便于溯源。
    返回：无（结果写库）。
    副作用：调用 ChatAgent 与 Judge 模型（耗时、耗配额），多次写 EvalRun.result_json。
    错误处理：
        异常不会抛给任何调用方（后台任务无处可抛），而是把结果改写成 status="error"，
        并把异常类型+信息塞进 ragas.error，让前端能直接看到失败原因。
    """
    # 后台任务在响应返回之后才执行，此时请求级的 db 已经关闭，必须自建 session。
    db = SessionLocal()
    # 先让出一次事件循环：确保 Starlette 已把「queued」响应发出去，
    # 否则后面的重活（导入、加载索引、初始化模型）会把客户端晾在那儿却拿不到 eval_id。
    await asyncio.sleep(0)

    async def persist_progress(progress: dict) -> None:
        """进度回调：每个阶段把整块 result_json 覆盖一遍并提交。

        用「覆盖」而不是增量更新：前端只需要拿到最新快照，覆盖逻辑最简单，
        也避免并发阶段之间字段互相残留。stage/message 之外的键原样并入 progress。
        """
        eval_run = db.get(EvalRun, UUID(eval_id))
        if eval_run is None:
            # eval 可能已被用户删掉，静默返回即可，不要因此中断整个评测。
            return
        eval_run.result_json = _job_result(
            status="running",
            stage=str(progress.get("stage") or "running"),
            message=str(progress.get("message") or "评测正在后台运行。"),
            repo_id=repo_id,
            evaluation_set=evaluation_set,
            **{
                key: value
                for key, value in progress.items()
                if key not in {"stage", "message"}
            },
        )
        db.add(eval_run)
        db.commit()

    try:
        await persist_progress({"stage": "starting", "message": "后台任务已启动，正在加载仓库上下文。"})
        result = await run_rag_quality_eval(
            db,
            repo_id,
            cases=cases,
            limit=limit,
            run_judge=run_judge,
            # 把进度回调交给 runner，它每完成一题/一个阶段就会回写一次。
            progress_callback=persist_progress,
            evaluation_set=evaluation_set,
        )
        result["progress"] = {
            "stage": "completed",
            "message": "评测已完成。",
            "updated_at": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        # logger.exception 带完整堆栈，便于事后定位；对外的信息只保留类型+消息。
        logger.exception("RAGAS background evaluation failed: eval_id=%s", eval_id)
        result = _job_result(
            status="error",
            stage="error",
            message=f"评测运行失败：{exc}",
            repo_id=repo_id,
            evaluation_set=evaluation_set,
        )
        # 补齐与成功结果相同的字段结构，前端不用为失败态写特判。
        result.update(
            {
                "passed": False,
                "ragas": {
                    "requested": run_judge,
                    "available": False,
                    "passed": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "metrics": {},
                },
                "cases": [],
            }
        )

    try:
        eval_run = db.get(EvalRun, UUID(eval_id))
        if eval_run is not None:
            eval_run.result_json = result
            db.add(eval_run)
            db.commit()
    finally:
        # 无论写库成败都要关 session，否则连接会泄漏（后台任务没有依赖注入兜底）。
        db.close()


def _job_result(
    *,
    status: str,
    stage: str,
    message: str,
    repo_id: str | None = None,
    evaluation_set: dict | None = None,
    **progress: object,
) -> dict:
    """构造异步任务的统一结果骨架（排队中/运行中/失败都用它）。

    入参：
        status / stage: 任务状态与所处阶段。
        repo_id: 目标仓库，决定 target 里有没有 repo_id。
        evaluation_set: 评测集元信息。
        **progress: 额外的进度字段，会并入 progress 对象。
    返回：结果 dict。
    说明：
        1. status 与 execution_status 同时存在是历史遗留（前端两个版本分别读这两个键），
           保留是为了不做破坏性改动；
        2. quality_status 默认 "incomplete"、evaluation_complete=False：只有评测真正跑完
           并由 runner 覆盖这些字段，才算「质量结果可信」，避免前端把排队中误判成通过；
        3. 所有键用关键字传入（* 强制），调用处一眼能看出每个值的含义。
    """
    return {
        "kind": "chatagent_ragas",
        "status": status,
        "execution_status": status,
        "quality_status": "incomplete",
        "evaluation_complete": False,
        "passed": False,
        "target": {"name": "ChatAgent", "repo_id": repo_id} if repo_id else {"name": "ChatAgent"},
        "evaluation_set": evaluation_set or {},
        "progress": {
            "stage": stage,
            "message": message,
            "updated_at": datetime.now(UTC).isoformat(),
            **progress,
        },
    }


@router.post("/{eval_id}/rescore", response_model=EvalRunResponse)
async def rescore_eval(
    eval_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> EvalRunResponse:
    """POST /api/evals/{eval_id}/rescore：只重跑「打分」，不重跑回答。

    入参：eval_id 为原评测 ID。
    返回：EvalRunResponse，指向**新建的**一条 EvalRun（status="queued"），
        原评测不会被修改——保留现场才能做前后对比。
    异常：
        原评测不存在 → 404；
        原结果里没有逐题现场 → 400（没东西可重评）；
        原结果没有失败/缺失的 Judge 指标 → 400（重评分没有意义）。
    副作用：新建一条 EvalRun 并排后台任务。
    为什么需要它：Judge 调用常因限流/超时失败，但 ChatAgent 的回答和检索结果都很贵，
        重跑一遍不划算；复用已存现场只重算指标，成本低一个量级。
    """
    source = db.get(EvalRun, eval_id)
    if source is None:
        raise HTTPException(status_code=404, detail="未找到 Eval run")
    source_result = source.result_json or {}
    if not source_result.get("cases"):
        raise HTTPException(status_code=400, detail="原评测没有可用于重评分的逐题现场")
    # 两类「可重评」的情形：有 Judge 报错，或有指标缺失。
    error_count = sum(bool(item.get("ragas_errors")) for item in source_result.get("cases") or [])
    missing_count = sum(
        any(name not in (item.get("ragas") or {}) for name in ("context_precision", "context_recall", "faithfulness", "answer_relevancy", "agent_goal_accuracy"))
        for item in source_result.get("cases") or []
    )
    if error_count == 0 and missing_count == 0:
        raise HTTPException(status_code=400, detail="原评测没有失败或缺失的 Judge 指标")
    repo_id = str((source_result.get("target") or {}).get("repo_id") or "")
    # 复制一份评测集元信息（浅拷贝即可），重评分结果里要标注「基于哪套题」。
    evaluation_set = dict(source_result.get("evaluation_set") or {})
    queued = _job_result(
        status="queued",
        stage="queued",
        message=f"已提交重评分任务，将复用 Eval {str(eval_id)[:8]} 的 ChatAgent 现场。",
        repo_id=repo_id or None,
        evaluation_set=evaluation_set,
        # rescore_of 标明血缘关系，前端据此显示「由哪次评测重评而来」。
        rescore_of=str(eval_id),
    )
    target = EvalRun(name=f"rescore-{source.name}", result_json=queued)
    db.add(target)
    db.commit()
    db.refresh(target)
    background_tasks.add_task(
        _execute_rescore_job,
        str(target.id),
        str(source.id),
        source_result,
        repo_id,
        evaluation_set,
    )
    return EvalRunResponse(eval_id=str(target.id), status="queued", result=queued)


async def _execute_rescore_job(
    target_eval_id: str,
    source_eval_id: str,
    source_result: dict,
    repo_id: str,
    evaluation_set: dict,
) -> None:
    """后台执行重评分：把原现场交给 rescore_rag_result，只补算指标。

    入参：
        target_eval_id: 新建的那条 EvalRun（写结果用）。
        source_eval_id: 原评测 ID（仅用于记录血缘）。
        source_result: 原评测的完整结果 JSON（含逐题现场）。
    副作用：调用 Judge 模型、多次写 target 的 result_json。
    错误处理：与评测任务一致——异常收敛成 status="error"，并把原 cases 原样保留，
        这样即使重评分失败，用户仍能看到原来的逐题结果。
    """
    db = SessionLocal()
    # 同样先让出事件循环，保证 queued 响应先到达客户端。
    await asyncio.sleep(0)

    async def persist_progress(progress: dict) -> None:
        """进度回调：覆盖写 target 的结果骨架（stage/message 之外的键并入 progress）。"""
        row = db.get(EvalRun, UUID(target_eval_id))
        if row is None:
            return
        row.result_json = _job_result(
            status="running",
            stage=str(progress.get("stage") or "rescore"),
            message=str(progress.get("message") or "正在重试异常评分。"),
            repo_id=repo_id or None,
            evaluation_set=evaluation_set,
            rescore_of=source_eval_id,
            **{key: value for key, value in progress.items() if key not in {"stage", "message"}},
        )
        db.add(row)
        db.commit()

    try:
        result = await rescore_rag_result(
            source_result,
            eval_id=source_eval_id,
            progress_callback=persist_progress,
        )
        result["progress"] = {
            "stage": "completed",
            "message": "异常指标重评分已完成；ChatAgent 回答和检索现场均未重新运行。",
            "updated_at": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        logger.exception("RAGAS rescore failed: eval_id=%s", target_eval_id)
        result = _job_result(
            status="error",
            stage="error",
            message=f"重评分失败：{exc}",
            repo_id=repo_id or None,
            evaluation_set=evaluation_set,
            rescore_of=source_eval_id,
        )
        result["ragas"] = {
            "requested": True,
            "available": False,
            "complete": False,
            "passed": None,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": {},
        }
        # 关键：把原来的逐题现场带回去，失败也不丢失历史结果。
        result["cases"] = source_result.get("cases") or []
    try:
        row = db.get(EvalRun, UUID(target_eval_id))
        if row is not None:
            row.result_json = result
            db.add(row)
            db.commit()
    finally:
        db.close()


def _eval_comparison(base: EvalRun, candidate: EvalRun) -> dict:
    """对比两次评测，产出「指标差值 + 逐题进退 + 能否发布」的报告。

    入参：base 为基线（通常是改动前），candidate 为候选（改动后）。
    返回：dict，含 comparable / incompatibility_reasons / metric_deltas /
        case_deltas / improved_cases / regressed_cases / release_gate_passed。
    说明：
        1. 先做「可比较性」检查：评测集 hash、Ragas 版本、Judge 模型、top_k 任一不同，
           结果就没有可比性。此时照样返回差值和逐题对比（供参考），
           但 comparable=False 且 release_gate_passed 一定为 False——这是防自欺的关键；
        2. 只有 base 和 candidate 都存在且是数值的指标才参与差值计算，避免 None 报错；
        3. 逐题「通过」的判定 = 硬规则通过 且 没有 Judge 错误；
        4. 取并集遍历，保证「只在某一边出现的用例」也能被识别为 improved/regressed。
    """
    base_result = base.result_json or {}
    candidate_result = candidate.result_json or {}
    base_set = base_result.get("evaluation_set") or {}
    candidate_set = candidate_result.get("evaluation_set") or {}
    reasons: list[str] = []
    if base_set.get("sha256") != candidate_set.get("sha256"):
        reasons.append("评测集 Hash 不一致")
    if (base_result.get("ragas") or {}).get("version") != (candidate_result.get("ragas") or {}).get("version"):
        reasons.append("Ragas 版本不一致")
    if (base_result.get("ragas") or {}).get("judge_model") != (candidate_result.get("ragas") or {}).get("judge_model"):
        reasons.append("Judge 模型不一致")
    if (base_result.get("pipeline") or {}).get("top_k") != (candidate_result.get("pipeline") or {}).get("top_k"):
        reasons.append("Top-K 不一致")
    base_metrics = base_result.get("summary_metrics") or {}
    candidate_metrics = candidate_result.get("summary_metrics") or {}
    metric_deltas = []
    # 取并集：新增/消失的指标也要出现在报告里，而不是被悄悄忽略。
    for name in sorted(set(base_metrics) | set(candidate_metrics)):
        left = base_metrics.get(name)
        right = candidate_metrics.get(name)
        # 布尔值在 Python 里也是 int 的实例，但指标不会是 bool，这里类型不匹配就跳过。
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            continue
        metric_deltas.append(
            {"name": name, "base": float(left), "candidate": float(right), "delta": round(float(right) - float(left), 4)}
        )
    base_cases = {str(item.get("id")): item for item in base_result.get("cases") or []}
    candidate_cases = {str(item.get("id")): item for item in candidate_result.get("cases") or []}
    case_deltas = []
    for case_id in sorted(set(base_cases) | set(candidate_cases)):
        left = base_cases.get(case_id) or {}
        right = candidate_cases.get(case_id) or {}
        # 通过 = 硬规则过 且 无 Judge 错误；缺数据时按「未通过」处理（保守）。
        left_passed = bool((left.get("hard_rules") or {}).get("passed")) and not bool(left.get("ragas_errors"))
        right_passed = bool((right.get("hard_rules") or {}).get("passed")) and not bool(right.get("ragas_errors"))
        state = "unchanged"
        if not left_passed and right_passed:
            state = "improved"
        elif left_passed and not right_passed:
            state = "regressed"
        case_deltas.append(
            {"id": case_id, "base_passed": left_passed, "candidate_passed": right_passed, "state": state}
        )
    return {
        "base_eval_id": str(base.id),
        "candidate_eval_id": str(candidate.id),
        # comparable=True 才允许做「结论性」判断。
        "comparable": not reasons,
        "incompatibility_reasons": reasons,
        "metric_deltas": metric_deltas,
        "case_deltas": case_deltas,
        "improved_cases": sum(item["state"] == "improved" for item in case_deltas),
        "regressed_cases": sum(item["state"] == "regressed" for item in case_deltas),
        "candidate_quality_status": candidate_result.get("quality_status"),
        "candidate_complete": bool(candidate_result.get("evaluation_complete")),
        # 发布门禁 = 可比较 且 候选评测整体通过；注意这里**没有**检查「是否有回归用例」，
        # 有回归但仍整体通过时门禁依然为 True，需要调用方自行参考 regressed_cases。
        "release_gate_passed": bool(not reasons and candidate_result.get("passed")),
    }


@router.get("/compare")
async def compare_evals(
    base_eval_id: UUID = Query(...),
    candidate_eval_id: UUID = Query(...),
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> dict:
    """GET /api/evals/compare?base_eval_id=...&candidate_eval_id=...：对比两次评测。

    入参：两个必填查询参数（缺任一个 → 422）。
    返回：_eval_comparison 的完整报告。
    异常：任一评测不存在 → 404。
    副作用：只读。
    注意：这个静态路径必须声明在 GET /{eval_id} 之前，否则 "compare" 会被当成 eval_id。
    """
    base = db.get(EvalRun, base_eval_id)
    candidate = db.get(EvalRun, candidate_eval_id)
    if base is None or candidate is None:
        raise HTTPException(status_code=404, detail="未找到用于对比的 Eval run")
    return _eval_comparison(base, candidate)


def _issue_markdown(eval_run: EvalRun) -> str:
    """把一次失败的评测渲染成 Issue 正文（Markdown）。

    入参：eval_run 为评测记录。
    返回：可直接贴进 Issue 的 Markdown 文本。
    说明：
        结构刻意固定为「复现环境 → 聚合结果 → 失败用例 → 验收标准」：
        报质量问题时最缺的就是复现信息，把这些（评测集 hash、Ragas 版本、Judge 模型）
        写死在正文里，接手的人不用再问一遍。
        用例只列「有诊断结论的」或「Judge 报错的」——通过的用例对排障没价值。
    """
    result = eval_run.result_json or {}
    suite = result.get("evaluation_set") or {}
    ragas = result.get("ragas") or {}
    lines = [
        "## 复现环境",
        "",
        f"- Eval ID：`{eval_run.id}`",
        f"- 评测集：`{suite.get('name') or '-'}`",
        f"- 评测集 Hash：`{suite.get('sha256') or '-'}`",
        f"- Ragas：`{ragas.get('version') or '-'}`",
        f"- Judge：`{ragas.get('judge_model') or '-'}`",
        f"- 运行状态：`{result.get('status') or '-'}`",
        f"- 质量状态：`{result.get('quality_status') or '-'}`",
        "",
        "## 聚合结果",
        "",
        "| 指标 | 分数 | 门槛 |",
        "| --- | ---: | ---: |",
    ]
    for name, detail in (ragas.get("metrics") or {}).items():
        # 只列门禁指标：其它辅助指标会让表格变长，反而看不清关键项。
        if name not in REQUIRED_RAGAS_METRICS:
            continue
        score = detail.get("score")
        threshold = detail.get("threshold")
        # 分数为 None 说明该项计算失败（通常是 Judge 报错），显式写成 ERROR 而不是留空。
        lines.append(
            f"| {name} | {score if score is not None else 'ERROR'} | {threshold if threshold is not None else '-'} |"
        )
    lines.extend(["", "## 失败用例", ""])
    for case in result.get("cases") or []:
        # diagnoses 里 category="passed" 的是「一切正常」，要过滤掉。
        diagnoses = [item for item in case.get("diagnoses") or [] if item.get("category") != "passed"]
        if not diagnoses and not case.get("ragas_errors"):
            continue
        lines.extend(
            [
                f"### {case.get('id') or 'unknown'}",
                "",
                f"- 问题：{case.get('question') or '-'}",
                f"- 标准答案：{case.get('reference') or '-'}",
                f"- 实际答案：{case.get('response') or '-'}",
            ]
        )
        if case.get("ragas_errors"):
            # 结构化错误原样贴出（JSON），便于直接复制检索或贴回日志系统。
            lines.append(f"- Judge 错误：`{json.dumps(case['ragas_errors'], ensure_ascii=False)}`")
        for diagnosis in diagnoses:
            lines.append(f"- {diagnosis.get('title')}：{diagnosis.get('action')}")
        lines.append("")
    lines.extend(
        [
            "## 验收标准",
            "",
            "- 使用同一评测集版本重新运行。",
            "- Judge 指标计算零错误。",
            "- 所有硬规则通过，所有门禁指标达到当前阈值。",
            "- 与本 Eval 做逐题对比，不新增回归用例。",
        ]
    )
    return "\n".join(lines)


@router.post("/{eval_id}/issue-draft")
async def create_eval_issue_draft(
    eval_id: UUID,
    payload: EvalIssueDraftRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> dict:
    """POST /api/evals/{eval_id}/issue-draft：把评测失败转成 Issue 草稿。

    入参：
        eval_id: 评测 ID。
        payload.title: 可选标题，不传则按「RAG 质量回归：{评测集} 未通过」生成。
        user: 需要 draft:create 权限；同时记录为草稿创建人。
    返回：{"draft_id", "status", "title"}，status 此时为 pending_confirmation。
    异常：
        评测不存在 → 404；
        评测里没存 repo_id（老数据）→ 400；
        评测对应仓库已被删 → 404。
    副作用：写一条 ActionDraft + 一条审计日志。
    安全设计：
        这里**不直接调 GitHub 建 issue**，而是生成 action_draft 交给人工确认，
        与 action_drafts.py 的 confirm 流程复用同一套审批链路。
    """
    eval_run = db.get(EvalRun, eval_id)
    if eval_run is None:
        raise HTTPException(status_code=404, detail="未找到 Eval run")
    result = eval_run.result_json or {}
    # repo_id 藏在 result["target"] 里（JSON 列），不是独立列，所以要层层取值。
    repo_id = (result.get("target") or {}).get("repo_id")
    if not repo_id:
        raise HTTPException(status_code=400, detail="该历史评测没有保存仓库信息，无法创建 Issue 草稿")
    repo = db.get(Repository, UUID(str(repo_id)))
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到评测对应仓库")
    suite_label = (result.get("evaluation_set") or {}).get("label") or "RAGAS"
    title = payload.title or f"RAG 质量回归：{suite_label} 未通过"
    draft = ActionDraft(
        repo_id=repo.id,
        draft_type="create_issue",
        target_type="repository",
        title=title,
        content=_issue_markdown(eval_run),
        # 风险定级 medium：建 issue 不会改代码，但会对外产生可见影响。
        risk_level="medium",
        created_by=user.id,
        # meta 里留 eval_id 和当时的质量状态，确认执行时能追溯来源。
        meta={"eval_id": str(eval_run.id), "quality_status": result.get("quality_status")},
    )
    db.add(draft)
    write_audit_log(
        db,
        user=user,
        repo_id=repo.id,
        action="eval.issue_draft.create",
        target_type="eval_run",
        target_id=str(eval_run.id),
        request_json={"eval_id": str(eval_run.id), "title": title},
    )
    db.commit()
    db.refresh(draft)
    return {"draft_id": str(draft.id), "status": draft.status, "title": draft.title}


@router.get("")
async def list_evals(
    limit: int = 20,
    repo_id: UUID | None = None,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> list[dict]:
    """GET /api/evals?limit=&repo_id=：评测历史列表。

    入参：
        limit: 默认 20，**这里是普通参数而不是 Query(...)**，所以没有 ge/le 校验，
            最终会被 max(1, min(limit, 100)) 夹到 1~100。
        repo_id: 可选，按仓库过滤。
    返回：列表项含 eval_id / name / status / created_at / metrics / 完整 result。
    异常：无。
    副作用：只读。
    实现说明：
        先固定取最近 100 条，再在**内存里**按 repo_id 过滤——因为 repo_id 存在
        result_json 里，SQL 层过滤不了。这也意味着「某仓库的评测如果不在最近 100 条内就查不到」，
        属于已知限制。
    """
    rows = db.query(EvalRun).order_by(EvalRun.created_at.desc()).limit(100).all()
    if repo_id is not None:
        rows = [
            row
            for row in rows
            if str(((row.result_json or {}).get("target") or {}).get("repo_id") or "") == str(repo_id)
        ]
    rows = rows[: max(1, min(limit, 100))]
    return [
        {
            "eval_id": str(row.id),
            "name": row.name,
            # 老数据没有 status 字段，兜底成 completed（当时只有同步评测，跑完才落库）。
            "status": str((row.result_json or {}).get("status") or "completed"),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "metrics": _summary_metrics(row.result_json or {}),
            "result": row.result_json,
        }
        for row in rows
    ]


@router.get("/{eval_id}")
async def get_eval(
    eval_id: str,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> dict:
    """GET /api/evals/{eval_id}：查询单个评测结果（前端轮询进度就靠它）。

    入参：eval_id 声明为 **str** 而不是 UUID，是为了自己控制错误码：
        格式不对时返回 400 而不是 FastAPI 默认的 422，前端提示更友好。
    返回：{"eval_id", "name", "status", "result"}。
    异常：ID 不是合法 UUID → 400；不存在 → 404。
    副作用：只读。
    """
    try:
        eval_uuid = UUID(eval_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Eval id 无效") from exc
    eval_run = db.get(EvalRun, eval_uuid)
    if eval_run is None:
        raise HTTPException(status_code=404, detail="未找到 Eval run")
    result = eval_run.result_json or {}
    return {
        "eval_id": str(eval_run.id),
        "name": eval_run.name,
        "status": str(result.get("status") or "completed"),
        "result": result,
    }


def _summary_metrics(result: dict) -> dict[str, float]:
    """从结果里抽出「可直接画图」的数值指标。

    入参：完整结果 dict。
    返回：{指标名: 浮点值}。
    说明：
        优先读 summary_metrics（标准结构）；没有的话退回扫描整个 result 的顶层字段，
        兼容早期版本把指标平铺在根上的老数据。
        显式排除 bool：Python 里 True/False 也是 int，会把「是否通过」当成 1.0/0.0
        混进指标里，导致图表出现莫名奇妙的 0/1 曲线。
    """
    summary = result.get("summary_metrics")
    metrics = {
        key: float(value)
        for key, value in (summary.items() if isinstance(summary, dict) else result.items())
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return metrics
