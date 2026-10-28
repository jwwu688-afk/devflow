"""Skill 注册中心：从磁盘加载 SKILL.md，按用户意图激活，并注入提示词。

「Skill」在本项目里指的是一份带 frontmatter 的 Markdown 说明书（backend/skills/<名字>/SKILL.md），
里面声明了入口工具名、可用工具、工作流步骤、安全级别和详细操作指引。

它解决的问题是：不要把每个 Agent 的能力都硬编码在代码里。运营/研发改一份 Markdown
就能新增或调整一个技能，不需要发版。

整条链路是：
    磁盘扫描 SKILL.md
      → 解析 frontmatter 得到 SkillManifest（强类型，字段缺失即报错）
      → 建立 name / entrypoint 两个索引
      → 运行时按「用户显式点名」或「触发词打分」选出 SkillActivation
      → 把 instructions + 被引用的资源文件拼成一段提示词塞进 LLM 上下文

加载是**启动时一次性**完成的（构造函数里全量读盘），之后只读内存。
因此改 SKILL.md 需要重启进程才生效；换来的是每次请求零 IO 开销。
"""

import ast
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SkillRegistryError(ValueError):
    """Skill 定义或激活过程中出现的、需要让调用方感知的配置错误。

    继承 ValueError 而非 Exception：语义上这确实是「参数/配置值不合法」，
    而且上层已有的 `except ValueError` 兜底逻辑（如 JSON 解析）可以顺带覆盖它，
    不必为它单独加一条捕获分支。
    """
    pass


class SkillManifest(BaseModel):
    """SKILL.md frontmatter 的强类型映射。

    只有 name/title/version/description/entrypoint 是必填，其余都有默认值，
    这样写一份最简 SKILL.md 也能跑起来，复杂技能再逐步补字段。
    字段类型校验在这里完成——frontmatter 写错格式会在**加载阶段**就炸，
    而不是等到运行时才发现某个技能行为异常。
    """

    name: str
    title: str
    version: str
    description: str
    category: str = "general"
    # entrypoint 是「工具名」，即该 Skill 绑定到哪个可调用工具上，
    # 用来建立 tool_name → Skill 的反向查找（见 _by_entrypoint）
    entrypoint: str
    # 以下四个列表都用 default_factory：Pydantic 中可变默认值必须显式工厂，
    # 否则所有实例会共享同一个 list 对象
    tools: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    # triggers 是自动激活的匹配依据，见 _activation_score
    triggers: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)
    # output_contract 描述输出格式契约（如 json / freeform），
    # safety_level 描述副作用等级（默认 read_only），
    # 两者都会被写进提示词，让模型知道边界
    output_contract: str = "freeform"
    safety_level: str = "read_only"


class SkillDefinition(BaseModel):
    """一份加载完成的 Skill：清单 + 文件路径 + 正文说明。"""

    manifest: SkillManifest
    # 用 str 而不是 Path，是为了能直接塞进 Pydantic 序列化结果 / 日志 / trace
    path: str
    # SKILL.md 去掉 frontmatter 之后的 Markdown 正文，最终会进入提示词
    instructions: str

    def trace_metadata(self) -> dict[str, Any]:
        """产出可并入链路追踪（trace）的扁平字段。

        只放定位用的轻量字段，不放 instructions 全文——否则每条 trace 都会
        被几 KB 的提示词撑爆。
        """
        return {
            "skill_name": self.manifest.name,
            "skill_title": self.manifest.title,
            "skill_version": self.manifest.version,
            "skill_steps": self.manifest.workflow_steps,
        }

    def prompt_summary(self) -> str:
        """生成一行式摘要，用于「可用技能清单」注入系统提示词。

        这段文本直接决定模型知不知道某个技能存在，所以刻意压成单行、
        带上 version 与安全的级别；重活细节留给被激活后的 prompt_block。
        """
        # 没写 workflow_steps 时给一句中文兜底，避免出现「步骤：」后面空着
        steps = ", ".join(self.manifest.workflow_steps) if self.manifest.workflow_steps else "在 SKILL.md 中定义"
        return (
            f"- {self.manifest.name} v{self.manifest.version} -> {self.manifest.entrypoint}: "
            f"{self.manifest.description} 步骤：{steps}。安全级别：{self.manifest.safety_level}。"
        )


class SkillActivation(BaseModel):
    """一次「技能被选中」的完整快照，也是最终拼进提示词的数据源。

    与 SkillDefinition 的区别：Definition 是静态定义，Activation 是**本次请求**
    的实例，额外记录了激活方式（explicit/automatic/entrypoint）、原因、
    以及本次实际加载进来的资源文件。
    """

    skill_name: str
    skill_title: str
    skill_version: str
    entrypoint: str
    tools: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)
    output_contract: str
    safety_level: str
    # 三选一：explicit（用户点名）、automatic（触发词命中）、entrypoint（专用入口直接调用）
    activation_mode: str
    activation_reason: str
    instructions: str
    # 指令正文的短摘要（sha256 前 16 位）。用途是判断「这次用的 Skill 版本内容
    # 是否和上次一致」，既便于回溯又不必落库全文
    instruction_digest: str
    resources: list[dict[str, Any]] = Field(default_factory=list)

    def trace_metadata(self) -> dict[str, Any]:
        """并入 trace 的字段，比 Definition 多出激活模式/原因/指令摘要。"""
        return {
            "skill_name": self.skill_name,
            "skill_title": self.skill_title,
            "skill_version": self.skill_version,
            "skill_steps": self.workflow_steps,
            "skill_activation_mode": self.activation_mode,
            "skill_activation_reason": self.activation_reason,
            "skill_instruction_digest": self.instruction_digest,
        }

    def prompt_block(self) -> str:
        """渲染成最终注入 LLM 的 Markdown 段落。

        结构：标题 → 激活方式与原因 → 允许工具 → 输出契约与安全级别 → 正文 → 资源。
        把「允许工具」和「安全级别」放在正文之前，是为了让模型先看到边界
        再看具体步骤，降低越权调用工具的概率。

        资源文件是**全量内联**的（不是按需加载）：因为 LLM 无法自己再去读盘，
        所谓「按需」只能在拼提示词时决定要不要带，一旦带上就是全文。
        """
        resource_text = ""
        if self.resources:
            sections = []
            for resource in self.resources:
                content = str(resource.get("content") or "").strip()
                # 二进制等非文本资源 content 为空串，只占位不内联，
                # 避免在提示词里插入一段空白噪声
                if content:
                    sections.append(f"### 资源：{resource['path']}\n{content}")
            if sections:
                resource_text = "\n\n按需加载的 Skill 资源：\n\n" + "\n\n".join(sections)
        return (
            f"## {self.skill_title} ({self.skill_name} v{self.skill_version})\n"
            f"激活方式：{self.activation_mode}；原因：{self.activation_reason}\n"
            f"允许工具：{', '.join(self.tools) or '无'}\n"
            f"输出契约：{self.output_contract}；安全级别：{self.safety_level}\n\n"
            f"{self.instructions.strip()}"
            f"{resource_text}"
        )


class SkillRegistry:
    """技能注册中心：负责加载、索引、匹配、激活。

    实例是不可变使用方式（加载后只读），所以可以安全地被多个请求共享；
    全局只通过 get_skill_registry() 拿同一个实例。
    """

    def __init__(self, root: Path) -> None:
        """扫描 root 下所有一级子目录里的 SKILL.md 并建立索引。

        执行顺序有依赖，不能调换：
        1) 先 _load_skills 读盘（此时还没有任何索引）；
        2) 再 _validate_unique_keys 校验 name / entrypoint 唯一，
           必须在建字典**之前**做——如果直接建字典，重复 key 会静默覆盖，
           问题会拖到运行期才以「某个技能神秘消失」的形式暴露；
        3) 最后建两个查找表。
        """
        self.root = root
        self._skills = self._load_skills()
        self._validate_unique_keys()
        # 两个索引：按技能名（供显式指定/管理接口用）和按入口工具名（供工具调用时反查）
        self._by_name = {skill.manifest.name: skill for skill in self._skills}
        self._by_entrypoint = {skill.manifest.entrypoint: skill for skill in self._skills}

    def list_skills(self) -> list[SkillDefinition]:
        """按名称排序返回全部技能。

        排序而非直接返回加载顺序：glob 的结果受文件系统影响，
        排序后提示词里的技能清单顺序稳定，便于缓存与 diff。
        """
        return sorted(self._skills, key=lambda skill: skill.manifest.name)

    def get_skill(self, name: str) -> SkillDefinition | None:
        """按技能名精确查找，找不到返回 None（不抛异常，便于调用方做可选逻辑）。"""
        return self._by_name.get(name)

    def skill_for_entrypoint(self, tool_name: str) -> SkillDefinition | None:
        """按入口工具名反查技能。工具执行前后用它补上 skill 相关的 trace 字段。"""
        return self._by_entrypoint.get(tool_name)

    def trace_metadata_by_tool(self) -> dict[str, dict[str, Any]]:
        """一次性导出「工具名 → trace 字段」的映射，用于 trace 初始化时预置。"""
        return {tool_name: skill.trace_metadata() for tool_name, skill in self._by_entrypoint.items()}

    def annotate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """给一次工具调用记录补上对应的 skill 元数据。

        找不到对应技能时返回**浅拷贝**而不是原对象：保持「返回值总是新 dict」
        这一约定，避免调用方改动返回值时意外污染入参。
        """
        tool_name = str(tool_call.get("tool_name") or "")
        skill = self.skill_for_entrypoint(tool_name)
        if not skill:
            return dict(tool_call)
        # 用 skill 的 trace 字段覆盖同名键：技能定义才是权威来源
        return {**tool_call, **skill.trace_metadata()}

    def prompt_context(self) -> str:
        """渲染「可用技能清单」，注入系统提示词让模型知道有哪些技能可用。

        空注册表要返回一句明确的中文而不是空串——空串会让模型误以为
        系统提示词拼装出错，进而胡乱编造技能名。
        """
        skills = self.list_skills()
        if not skills:
            return "当前没有注册 DevFlow 技能。"
        return "\n".join(skill.prompt_summary() for skill in skills)

    def tool_description_suffix(self, tool_name: str) -> str:
        """给工具描述追加技能说明，让模型在选择工具时看到技能信息。

        返回带前导空格的字符串，是给调用方直接做字符串拼接用的；
        没有对应技能时返回空串，拼接后无变化。
        """
        skill = self.skill_for_entrypoint(tool_name)
        if not skill:
            return ""
        return (
            f" 注册技能：{skill.manifest.title} v{skill.manifest.version}。"
            f"{skill.manifest.description}"
        )

    def tool_description_suffixes(self) -> dict[str, str]:
        """批量版本：一次拿到所有入口工具的说明后缀。"""
        return {tool_name: self.tool_description_suffix(tool_name) for tool_name in self._by_entrypoint}

    def activate(
        self,
        message: str,
        *,
        requested_name: str | None = None,
        max_auto_skills: int = 3,
    ) -> list[SkillActivation]:
        """根据用户消息决定激活哪些技能，返回按优先级排好的激活列表。

        两级策略，显式优先于自动：
        1) 显式：requested_name 指定，或消息里点名了某个技能 → 直接返回，不再打分。
           点名即视为强意图，哪怕触发词一个字都没命中。
        2) 自动：对所有技能算触发词匹配分，取分数 > 0 的前 max_auto_skills 个。

        max_auto_skills 默认 3：每激活一个技能就要往上下文里塞一大段
        instructions，数量不设上限会迅速吃掉预算、稀释重点。

        注意：显式指定但名字不存在时由 _explicit_skills 抛 SkillRegistryError，
        这里不做兜底——配错名字属于调用方 bug，静默降级成「无技能」更难排查。
        """
        explicit = self._explicit_skills(message, requested_name)
        if explicit:
            # 显式命中直接短路返回，不做任何打分与截断（不受 max_auto_skills 限制）
            return [self._activation(skill, "explicit", "用户明确指定 Skill") for skill in explicit]

        scored = [
            (self._activation_score(message, skill), skill)
            for skill in self._skills
        ]
        # 只为「至少命中一个触发词」的技能生成激活对象，0 分不进入上下文
        selected = [
            (score, skill)
            for score, skill in scored
            if score > 0
        ]
        # 先按分数降序，分数相同再按名称升序——保证同分时的入选结果可复现，
        # 不依赖 dict / glob 的遍历顺序
        selected.sort(key=lambda item: (-item[0], item[1].manifest.name))
        return [
            self._activation(skill, "automatic", f"消息命中 Skill 触发条件，匹配分 {score}")
            for score, skill in selected[:max_auto_skills]
        ]

    def activate_entrypoint(self, tool_name: str, *, reason: str = "专用 Agent 执行入口") -> SkillActivation | None:
        """由工具名直接激活（专用 Agent 走自己的入口时用），找不到返回 None。

        与 activate 的区别：这里不做任何意图匹配，调用方已经确切知道要用哪个技能。
        """
        skill = self.skill_for_entrypoint(tool_name)
        if skill is None:
            return None
        return self._activation(skill, "entrypoint", reason)

    def _activation(self, skill: SkillDefinition, mode: str, reason: str) -> SkillActivation:
        """把静态 SkillDefinition 转成一次性的 SkillActivation 快照。

        这一步会读盘加载被引用的资源文件（_load_referenced_resources），
        所以它不是纯函数；资源校验失败会抛 SkillRegistryError。
        """
        instructions = skill.instructions.strip()
        return SkillActivation(
            skill_name=skill.manifest.name,
            skill_title=skill.manifest.title,
            skill_version=skill.manifest.version,
            entrypoint=skill.manifest.entrypoint,
            tools=skill.manifest.tools,
            workflow_steps=skill.manifest.workflow_steps,
            output_contract=skill.manifest.output_contract,
            safety_level=skill.manifest.safety_level,
            activation_mode=mode,
            activation_reason=reason,
            instructions=instructions,
            # digest 基于 strip 后的正文计算，与 instructions 字段保持一致；
            # 取前 16 位（64 bit）够用且日志里不会太长
            instruction_digest=hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16],
            resources=self._load_referenced_resources(skill),
        )

    def _explicit_skills(self, message: str, requested_name: str | None) -> list[SkillDefinition]:
        """识别「用户明确指定了某个技能」的情况。

        三种判定路径，优先级从高到低：
        1) requested_name 参数（来自 API 显式字段）——最可靠，找不到直接报错；
        2) 消息里出现 "skill/技能" 且直接包含某个技能名或标题——宽松匹配，命中即返回；
        3) 正则兜底："用/使用/调用 XXX skill" 句式——此时若消息明确点名了技能
           却发现该名字不存在，抛错而不是静默忽略。

        第 2 步要求消息里先出现 "skill/技能" 字样才继续判断，目的是防止
        技能名恰好是常用词（如 "review"）时被误判为显式指定。
        """
        if requested_name:
            skill = self._skill_by_name_or_title(requested_name)
            if skill is None:
                raise SkillRegistryError(f"未找到用户指定的 Skill：{requested_name}")
            return [skill]

        # 统一转小写做包含匹配；中文没有大小写问题，所以 "技能" 直接在原串里找
        lowered = message.lower()
        if "skill" not in lowered and "技能" not in message:
            return []
        selected = [
            skill
            for skill in self._skills
            if skill.manifest.name.lower() in lowered or skill.manifest.title.lower() in lowered
        ]
        if selected:
            return selected

        # 兜底句式匹配：允许技能名是「英文标识符」形式。
        # 注意这里只捕获 [A-Za-z][A-Za-z0-9_-]+，纯中文技能名走不到这条分支
        match = re.search(
            r"(?:use|using|用|使用|调用|启用)\s*([A-Za-z][A-Za-z0-9_-]+)\s*(?:skill|技能)",
            message,
            flags=re.IGNORECASE,
        )
        if match:
            raise SkillRegistryError(f"未找到用户指定的 Skill：{match.group(1)}")
        return []

    def _skill_by_name_or_title(self, value: str) -> SkillDefinition | None:
        """按技能名或标题查找，双方 strip+lower 后做精确相等比较（不是包含匹配）。"""
        normalized = value.strip().lower()
        return next(
            (
                skill
                for skill in self._skills
                if skill.manifest.name.lower() == normalized or skill.manifest.title.lower() == normalized
            ),
            None,
        )

    def _activation_score(self, message: str, skill: SkillDefinition) -> int:
        """按触发词命中情况给技能打分。

        打分规则的取舍：
        - 英文/符号类触发词（如 "ci"、"pytest"、"c++"）必须做**词边界**匹配，
          否则 "ci" 会命中 "specific"、"decision" 这类词里的子串，造成大面积误激活；
          边界定义为「前后不是 [a-z0-9_]」，比 \\b 更严格（避免 c++ 这类符号被切断）。
        - 中文等含非 ASCII 的触发词直接做子串包含，因为中文没有空格分词。
        - 每命中一个词得 max(1, min(len, 8)) 分：长词更具体、更可信，给更高权重；
          但封顶 8 分，防止一个超长触发词单独压过其他所有信号。
        """
        lowered = message.lower()
        score = 0
        for trigger in skill.manifest.triggers:
            normalized = trigger.strip().lower()
            # 空触发词（frontmatter 里写了 "- " 之类）直接跳过，否则会无条件命中
            if not normalized:
                continue
            if re.fullmatch(r"[a-z0-9_+#.-]+", normalized):
                # 纯 ASCII 词 → 词边界匹配，避免上面说的子串误命中
                matched = re.search(rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])", lowered)
            else:
                # 含中文/其他字符 → 退化为包含匹配
                matched = normalized in lowered
            if matched:
                score += max(1, min(len(normalized), 8))
        return score

    def _load_referenced_resources(self, skill: SkillDefinition) -> list[dict[str, Any]]:
        """加载 SKILL.md 正文里用反引号引用的资源文件。

        只有形如 `references/xxx.md`、`templates/...`、`scripts/...`、`assets/...`
        的引用才会被识别，并且必须在**同一目录下的这四类子目录**里。

        这里做了一串硬性校验，逐条都是为了防「Skill 变成任意文件读取通道」：
        - 上限 8 个：防止一次激活就把上下文撑爆（每个文件最多 64 KiB）；
        - relative_to(skill_root)：挡住 `references/../../etc/passwd` 这种路径穿越；
        - is_file()：挡住指向目录或已被删除的引用；
        - 64 KiB 上限：单文件体积预算，8 × 64KiB ≈ 512KiB 是绝对上限；
        - 非文本后缀只记元数据不读内容：避免把二进制当文本塞进提示词。

        任何一条不满足都抛 SkillRegistryError——宁可让整个 Skill 加载失败，
        也不要带着一个「悄悄少读了文件」的技能去执行，那种问题极难发现。
        """
        # 用 set 去重：同一资源在正文里被引用两次只加载一次
        references = sorted(
            {
                match.group(1).strip()
                for match in re.finditer(
                    r"`((?:references|templates|scripts|assets)/[^`]+)`",
                    skill.instructions,
                )
            }
        )
        if len(references) > 8:
            raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源超过 8 个")

        # skill.path 指向 SKILL.md 本身，取 parent 才是技能目录
        skill_root = Path(skill.path).resolve().parent
        resources: list[dict[str, Any]] = []
        # 白名单式判定：只有这些后缀才读文本，其余（图片、pdf 等）content 留空
        text_suffixes = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".py", ".sh", ".ps1"}
        for relative in references:
            resource_path = (skill_root / relative).resolve()
            # resolve 之后再做包含性检查，才能真正挡住 ../ 穿越；
            # 若不在 skill_root 之下，relative_to 会抛 ValueError
            try:
                resource_path.relative_to(skill_root)
            except ValueError as exc:
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用了目录外资源：{relative}") from exc
            if not resource_path.is_file():
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源不存在：{relative}")
            if resource_path.stat().st_size > 64 * 1024:
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源超过 64 KiB：{relative}")
            content = resource_path.read_text(encoding="utf-8") if resource_path.suffix.lower() in text_suffixes else ""
            resources.append(
                {
                    # 统一成 / 分隔，避免 Windows 上生成 "references\\a.md" 与文档不一致
                    "path": relative.replace("\\", "/"),
                    "content": content,
                    "size": resource_path.stat().st_size,
                    # 资源摘要用全量 bytes 计算（不是 content），
                    # 这样二进制文件也有指纹可用于变更检测
                    "digest": hashlib.sha256(resource_path.read_bytes()).hexdigest()[:16],
                }
            )
        return resources

    def _load_skills(self) -> list[SkillDefinition]:
        """扫描 <root>/*/SKILL.md 并逐个解析。

        只认「一级子目录 + 文件名恰为 SKILL.md」的结构，不做递归：
        这样技能目录里放多少辅助资源都不会被误当成技能。

        root 不存在时返回空列表而不是报错——允许在没有任何技能的部署形态下启动。
        """
        if not self.root.exists():
            return []
        skills: list[SkillDefinition] = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skills.append(_load_skill(skill_file))
        return skills

    def _validate_unique_keys(self) -> None:
        """校验 name 与 entrypoint 各自全局唯一。

        必须在建索引字典之前调用：dict 推导式遇到重复 key 会静默保留后者，
        导致「同名技能随机生效一个」这种极难排查的问题。
        """
        seen_names: set[str] = set()
        seen_entrypoints: set[str] = set()
        for skill in self._skills:
            if skill.manifest.name in seen_names:
                raise SkillRegistryError(f"Skill 名称重复：{skill.manifest.name}")
            if skill.manifest.entrypoint in seen_entrypoints:
                raise SkillRegistryError(f"Skill entrypoint 重复：{skill.manifest.entrypoint}")
            seen_names.add(skill.manifest.name)
            seen_entrypoints.add(skill.manifest.entrypoint)


def _load_skill(path: Path) -> SkillDefinition:
    """读取单个 SKILL.md：拆 frontmatter → Pydantic 校验 manifest → 拼 Definition。

    frontmatter 解析失败抛 SkillRegistryError，字段缺失/类型错误则由
    Pydantic 抛 ValidationError，两者都会让整个注册中心初始化失败（fail fast）。
    """
    text = path.read_text(encoding="utf-8")
    manifest_data, instructions = _split_frontmatter(text, path)
    manifest = SkillManifest.model_validate(manifest_data)
    return SkillDefinition(manifest=manifest, path=str(path), instructions=instructions)


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    """把 Markdown 拆成 (frontmatter 字典, 正文)。

    格式约定：文件必须以单独一行 `---` 开头，以另一行 `---` 结束，
    中间是简易 YAML，之后全部是正文。

    先统一换行符：Windows 上编辑的文件常是 CRLF，不归一化会导致
    `line.strip() == "---"` 之外的行尾比较和 join 结果带上 \\r。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # 允许首行 `---` 前后有空白，但不允许 BOM/其他字符（有的话直接报错更易定位）
    if not lines or lines[0].strip() != "---":
        raise SkillRegistryError(f"{path} 必须以 frontmatter 开头")

    frontmatter_lines: list[str] = []
    body_start = None
    # 从第 2 行开始找结束标记；只认第一个 `---`，正文里的分隔线不会被误判
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            # body_start 指向结束标记的**下一行**（index 是 0 基，+1 即下一行）
            body_start = index + 1
            break
        frontmatter_lines.append(line)
    if body_start is None:
        raise SkillRegistryError(f"{path} 缺少 frontmatter 结束标记")

    # 正文整体 strip，避免开头连续空行被带进提示词
    return _parse_frontmatter(frontmatter_lines), "\n".join(lines[body_start:]).strip()


def _parse_frontmatter(lines: list[str]) -> dict[str, Any]:
    """手写的最小 YAML 子集解析器（不引入 PyYAML 依赖）。

    支持的语法仅四种，刚好覆盖 SKILL.md 的需要：
        key: value            # 标量
        key:                  # 空值 → 视为列表开始
          - item
        key: [a, b, c]        # 行内列表
        # 注释行

    为什么不用 PyYAML：这里的文件格式由我们自己控制，额外依赖只会带来
    「YAML 支持了但我们没处理」的隐式契约；而且 PyYAML 的安全反序列化限制
    需要额外注意。手写版遇到不认识的语法会直接报错，行为可预期。

    解析状态只有 current_list_key 一个：它记录「正在收集哪个 key 的列表项」，
    遇到新的 key 就重置。
    """
    data: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        # 跳过空行与整行注释
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            # 列表项必须依附于前一个空值 key，否则无从归属
            if current_list_key is None:
                raise SkillRegistryError("列表项缺少父级 key")
            data.setdefault(current_list_key, [])
            # 防止「先 key: 标量，后 - item」的混写：那种情况 data[key] 不是 list
            if not isinstance(data[current_list_key], list):
                raise SkillRegistryError(f"{current_list_key} 不是列表")
            data[current_list_key].append(_parse_value(stripped[2:]))
            continue
        if ":" not in raw_line:
            raise SkillRegistryError(f"frontmatter 行无效：{raw_line}")
        # split(":", 1) 只切第一个冒号：值里含冒号（如 URL、时间）时不会被截断
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SkillRegistryError("frontmatter key 不能为空")
        if value == "":
            # 空值先占位成空列表，并把它设为「当前列表」，
            # 后续的 "- item" 会 append 进来
            data[key] = []
            current_list_key = key
        else:
            data[key] = _parse_value(value)
            # 标量值结束列表上下文，避免后续 "- item" 被错误归到这个 key 下
            current_list_key = None
    return data


def _parse_value(value: str) -> Any:
    """把 frontmatter 里的标量字符串转成 Python 值。

    转换顺序有讲究：
    1) 空串 → ""（保留原样，交由 Pydantic 判断必填）；
    2) [..] → 先试 ast.literal_eval（正确处理引号、逗号、转义），
       失败再退化为朴素逗号分割——literal_eval 在遇到未加引号的中文时会抛
       SyntaxError，这种常见写法必须能兜住；
    3) true/false → 布尔（因为 Pydantic 需要真正的 bool，字符串 "true" 会校验失败）；
    4) 成对引号 → 去引号（支持中文里写 "含冒号的值"）；
    5) 其他 → 原样返回字符串。

    注意：所有 key 都走这个函数，所以 "20221128" 这类纯数字会保持字符串，
    不会被误转成 int。
    """
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            # 退化路径：[a, b] 这种没加引号的写法
            parsed = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
        # literal_eval 也可能解析出非 list（如字符串），此时保守地返回原值
        return parsed if isinstance(parsed, list) else value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


@lru_cache(maxsize=1)
def get_skill_registry() -> SkillRegistry:
    """进程内单例：首次调用时扫描磁盘，之后直接返回缓存实例。

    lru_cache(maxsize=1) 而不是模块级全局变量：这样既保证单例，
    又能在测试里用 get_skill_registry.cache_clear() 重置。

    root 的计算路径：registry.py 位于 backend/app/services/skills/，
    parents[0]=skills、[1]=services、[2]=app…… 再往上是 backend，
    但这里取 parents[2] 后直接拼 "skills"，即 backend/app/skills。
    """
    root = Path(__file__).resolve().parents[2] / "skills"
    return SkillRegistry(root)
