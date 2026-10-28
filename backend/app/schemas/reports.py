"""周报模型（业务对象：按时间区间生成的仓库研发周报）。

输出 Markdown 而不是结构化 JSON：周报的最终消费者是人（复制到文档/群里），
Markdown 能直接粘贴且保留排版；若前端需要图表，再从同一份数据源另行结构化即可。
"""

from datetime import date
from uuid import UUID

from pydantic import BaseModel


class WeeklyReportRequest(BaseModel):
    """生成周报的请求（业务对象：要统计的仓库 + 时间区间）。

    用 date 而不是 datetime：周报是「按天」粒度，时间部分没有意义，
    收 date 也能让前端日期选择器直接传 YYYY-MM-DD，省掉时区换算的歧义。
    """
    repo_id: UUID
    start_date: date  # 区间起（含）
    end_date: date  # 区间止（含）；由服务层校验 start <= end


class WeeklyReportResponse(BaseModel):
    """周报内容（业务对象：渲染好的 Markdown 正文）。"""
    report_markdown: str
