# MCP（Model Context Protocol）服务包入口。
# 本文件刻意保持「只有注释、没有代码」：具体能力由 memory_server 等子模块自行导出。
# 若在这里 import 子模块，会在包被首次引用时触发整个 MCP 依赖链的加载（含环境变量校验），
# 既拖慢启动，也容易与 app.core 形成循环导入，因此保持空实现。
