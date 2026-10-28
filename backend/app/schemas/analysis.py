"""AI 分析结果的结构化输出模型（业务对象：LLM 对 Issue / PR / CI 给出的分析结论）。

为什么要为每个分析场景定义严格的 schema：
这些类同时充当「LLM 结构化输出的契约」。它们会被转成 JSON Schema 交给模型做 function calling，
Literal 枚举把模型的自由发挥收敛到固定取值，confidence 用 ge/le 卡死 0~1，
从而避免下游统计时出现「priority 写成 high」「置信度写成 95」这类脏数据。
代价是：枚举一旦扩充（例如新增一种结论），prompt 与前端展示需要同步更新。
"""

from typing import Literal

from pydantic import BaseModel, Field


class DuplicateCandidate(BaseModel):
    """疑似重复的 Issue（业务对象：与当前 Issue 语义相近、可能应当合并的另一条 Issue）。"""
    issue_id: str  # 目标 Issue 的标识（字符串以兼容 GitHub 编号与内部 UUID）
    title: str  # 标题，供人工快速判断是否真的重复
    score: float = Field(ge=0, le=1)  # 相似度 0~1；卡范围是防止模型输出 95 这种「百分制」写法


class IssueEvidence(BaseModel):
    """分析所依据的一条证据（业务对象：支撑结论的代码片段/文档/历史 Issue）。"""
    source_type: str  # 证据类型：code / doc / issue / pr ...
    title: str  # 证据标题，前端折叠面板的标题行
    snippet: str  # 证据摘录正文
    reference: str | None = None  # 可定位的引用（文件路径:行号、URL）；允许为空
    confidence: float | None = Field(default=None, ge=0, le=1)  # 该条证据的可信度；None 表示未评估


class IssueDrafts(BaseModel):
    """可直接复用的草稿文本（业务对象：AI 代笔的评论与任务拆解，供人确认后发布）。"""
    clarification_comment: str | None = None  # 向提问者追问的评论草稿
    task_breakdown: str | None = None  # 任务拆解说明（子任务清单）


class IssueAnalysisOutput(BaseModel):
    """Issue 分析结论（业务对象：一条 Issue 的「分类 + 优先级 + 是否重复 + 行动建议」报告）。

    这是 issue 分析链路的核心输出，前端会按字段分区展示；
    conclusion 与 category 用中文枚举而不是英文码，是因为它们会被直接渲染给最终用户，
    省掉一层映射也就少了一处翻译不一致的风险。
    """
    summary: str  # 一句话摘要，列表页展示
    # 结论五选一：进入开发 / 等待澄清 / 关闭 / 拆分 / 合并到已有 Issue
    conclusion: Literal["进入开发", "等待澄清", "关闭", "拆分", "合并到已有 Issue"]
    conclusion_reason: str  # 给出结论的理由，便于人工复核（防止模型拍脑袋）
    category: Literal["Bug", "Feature", "Question", "Documentation", "Refactor", "Test", "Ops"]  # 问题类别
    priority: Literal["P0", "P1", "P2", "P3"]  # 优先级，P0 最高
    complexity: Literal["S", "M", "L", "XL"]  # 工作量档位，用于排期估算
    suggested_owner: str  # 建议负责人（通常是模块 owner 或最近提交者）
    owner_reason: str  # 推荐理由，避免「凭感觉指派」
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)  # 疑似重复项，可能为空
    evidence: list[IssueEvidence] = Field(default_factory=list)  # 支撑证据
    checklist: list[str] = Field(default_factory=list)  # 验收清单
    action_items: list[str] = Field(default_factory=list)  # 下一步行动项
    drafts: IssueDrafts = Field(default_factory=IssueDrafts)  # 草稿文本；用 factory 避免跨实例共享
    confidence: float = Field(ge=0, le=1)  # 整体置信度，低于阈值时前端会提示「建议人工复核」


class PRReviewFinding(BaseModel):
    """PR 评审中的一条问题（业务对象：需要作者处理的具体缺陷或改进点）。"""
    severity: Literal["P1", "P2", "P3"]  # 严重程度；这里没有 P0，因为 PR 阶段的问题不会立即造成线上事故
    title: str  # 问题概要
    evidence: str  # 证据（相关代码片段），没有证据的评审意见容易被作者反驳
    required_action: str  # 期望作者做的修改
    blocking: bool = False  # 是否阻塞合入；默认不阻塞，避免小问题卡住整个 PR


class PRReviewOutput(BaseModel):
    """PR 评审报告（业务对象：一次 AI Code Review 的完整结论）。

    plan / executed_steps 保留了「先计划再执行」的推理过程，
    既是审计依据，也便于定位「哪一步没做导致漏审」。
    """
    summary: str  # 变更概述
    plan: list[str] = []  # 评审计划（模型自述要看哪些文件、查什么）
    executed_steps: list[str] = []  # 实际执行的步骤，与 plan 对照可发现问题
    merge_recommendation: Literal["建议合入", "修改后合入", "暂缓", "拒绝"] = "暂缓"  # 合入建议；默认「暂缓」是最保守选择
    recommendation_reason: str = ""  # 建议理由
    review_findings: list[PRReviewFinding] = Field(default_factory=list)  # 结构化的具体问题
    key_changes: list[str] = []  # 关键变更点，帮助 reviewer 快速抓重点
    risk_points: list[str] = []  # 风险点（兼容性、性能、安全等）
    blocking_issues: list[str] = []  # 阻塞项
    review_checklist: list[str] = []  # 评审自查清单
    test_suggestions: list[str] = []  # 建议补充的测试
    files_need_attention: list[str] = []  # 需重点关注的文件
    review_comments: list[str] = []  # 可直接贴到 PR 的评论文本
    confidence: float = Field(ge=0, le=1)  # 整体置信度


class CIDebugOutput(BaseModel):
    """CI 失败诊断结论（业务对象：对一次失败流水线的根因分析 + 修复建议）。

    failure_type 用固定枚举，是为了让「失败类型分布」这类统计能直接做聚合，
    否则模型可能写出「测试挂了」「test failure」等五花八门的表述。
    """
    failure_summary: str  # 失败现象概述
    failure_type: Literal["test", "build", "lint", "dependency", "permission", "environment", "unknown"]  # 归因分类
    plan: list[str] = []  # 排查计划
    executed_steps: list[str] = []  # 实际排查步骤
    first_error: str | None = None  # 日志里的首个错误行，通常是最有价值的线索
    root_cause: str = ""  # 推断的根因
    possible_causes: list[str] = []  # 备选可能原因（根因不确定时的候选）
    fix_steps: list[str] = []  # 修复步骤
    debug_steps: list[str] = []  # 复现/验证步骤
    related_files: list[str] = []  # 相关文件
    # 默认 True：CI 挂了通常不该合入，宁可让用户手动放行，也不要默认放过
    is_merge_blocking: bool = True
    blocking_reason: str = ""  # 阻塞理由
    confidence: float = Field(ge=0, le=1)  # 置信度


class SafetyOutput(BaseModel):
    """安全判定结论（业务对象：某个拟执行动作是否需要人工确认、风险多高）。

    这是「人在环」的闸门：requires_confirmation 为真时，系统不会自动执行，
    而是把 draft 转成待确认的 ActionDraft 交给用户。
    """
    requires_confirmation: bool  # 是否必须人工确认
    risk_level: Literal["low", "medium", "high"]  # 风险等级；high 时前端会强制二次确认
    reason: str  # 判定理由，解释「为什么判定为这个风险」
    draft: str | None = None  # 建议的动作内容草稿；无需执行时为空
