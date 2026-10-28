"""skills 子包的对外出口。

把 registry 里的核心类型一次性导出，让调用方写 `from app.services.skills import SkillRegistry`
即可，不必关心内部模块拆分。__all__ 同时起到「公开 API 白名单」的作用：
以后 registry 内部再拆文件，只要这个出口不变，业务代码就不用改。
"""

from app.services.skills.registry import SkillActivation, SkillDefinition, SkillManifest, SkillRegistry, SkillRegistryError, get_skill_registry

__all__ = ["SkillActivation", "SkillDefinition", "SkillManifest", "SkillRegistry", "SkillRegistryError", "get_skill_registry"]
