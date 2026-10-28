"""日志初始化：进程启动时调用一次，统一全应用的日志格式与级别。

只做 basicConfig，因为它作用于 root logger，各模块直接 logging.getLogger(__name__) 即可，
无需层层传递 logger 实例；代价是无法按模块单独调级别（课程项目场景下可接受）。
"""

import logging


def configure_logging() -> None:
    """配置 root logger：INFO 级别 + 「时间 级别 [模块] 消息」格式。

    级别固定 INFO：既能看到关键流程，又不会把第三方库的 DEBUG 噪音刷满终端。
    [%(name)s] 用模块名而不是硬编码标签，方便按文件定位日志来源。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
