"""数据源路由策略：决定一份数据「进向量库检索」还是「直连查询」。

这是 RAG 链路最前面的一个分叉点，也是最容易被忽略、但影响最大的一步。
核心判断标准只有一条：**这份数据是不是「可复用的非结构化知识」**。

- 走 RAG（rag）：issue、PR、CI 运行记录、项目文档、知识文件、记忆笔记。
  这些内容体量大、需要语义召回，且写入后基本不再变化（或变化不敏感），
  适合先向量化、之后靠相似度找回来。

- 走 direct（直连）：项目概览、manifest、成员、周报、会话与线程记忆。
  这些是**结构化、强时效**的事实——成员名单今天改明天就过期，
  如果也塞进向量库，就会出现「检索到三天前的团队名单」这种硬伤。
  所以它们必须每次实时查源表/API。

后果：向量库因此不是「全量数据库」，而只是知识的一个可检索切片。
排查「为什么某条信息检索不到」时，第一步就该来这里确认它的 source_type
到底被路由到了哪一边。
"""

from collections.abc import Iterable
from typing import Literal


# RetrievalMode 用 Literal 而不是 Enum：它只出现在内部函数签名与字典值里，
# 不需要序列化/反射，Literal 更轻，且能被类型检查器当作字符串字面量收窄
RetrievalMode = Literal["rag", "direct"]

# RAG is reserved for reusable, unstructured project knowledge where semantic
# recall is useful. Structured live facts remain in their source tables/APIs.
# RAG 只用于「可复用的非结构化项目知识」，追求语义召回；
# 结构化且实时变化的事实留在源表/API，不进向量库。
# 用 frozenset 而非 set：这两个集合是模块级常量，必须防止被意外修改。
RAG_SOURCE_TYPES = frozenset(
    {
        "issue",
        "pull_request",
        "workflow_run",
        "project_doc",
        "knowledge_file",
        "memory_note",
    }
)

# 直连类型：内容会随时间/操作变化，向量化瞬间就过期
DIRECT_SOURCE_TYPES = frozenset(
    {
        "project_overview",
        "project_manifest",
        "team_member",
        "weekly_report",
        "chat_session",
        "conversation_memory",
        "thread_memory",
    }
)


def retrieval_mode(source_type: str) -> RetrievalMode:
    """判定某个 source_type 的检索方式。

    注意这里只查 RAG 白名单，其余一律回落 direct——**白名单优先于黑名单**，
    这样新增一个 source_type 而忘了配置时，默认走「实时直查」这条更安全的路径，
    不会因为漏配而悄悄把时效性数据冻进向量库。
    DIRECT_SOURCE_TYPES 目前只用于文档说明与外部校验，本函数并不依赖它。
    """
    if source_type in RAG_SOURCE_TYPES:
        return "rag"
    return "direct"


def normalize_source_types(source_types: Iterable[str | None] | None) -> list[str]:
    """清洗调用方传入的 source_types 列表。

    做三件事：转字符串并去首尾空白 → 丢弃空值 → 去重且**保持首次出现顺序**。
    用 dict.fromkeys 而不是 set，是为了让结果顺序稳定（set 顺序不确定，
    会影响后续缓存 key 与日志可读性）。

    入参为 None 时按空列表处理，避免调用方到处写判空。
    """
    normalized = [str(source_type).strip() for source_type in (source_types or []) if source_type and str(source_type).strip()]
    return list(dict.fromkeys(normalized))


def should_vectorize(source_type: str) -> bool:
    """写入侧判定：这条数据要不要生成向量并落向量库。"""
    return retrieval_mode(source_type) == "rag"


def is_searchable_source(source_type: str) -> bool:
    """读取侧判定：这个类型能不能作为向量检索的来源。

    与 should_vectorize 逻辑相同但语义不同——一个是写入时的取舍，
    一个是查询时的过滤。保留两个名字是为了让调用点自解释。
    """
    return retrieval_mode(source_type) == "rag"


def policy_metadata(source_type: str) -> dict[str, object]:
    """生成要随数据一起落库的策略元信息。

    把 retrieval_mode / rag_enabled 写进 metadata 而不是运行时再算：
    策略以后可能调整，落库时快照能保证「这批数据当时是按什么规则处理的」可追溯。
    """
    mode = retrieval_mode(source_type)
    return {
        "retrieval_mode": mode,
        "rag_enabled": mode == "rag",
    }
