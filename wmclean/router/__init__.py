"""L2 Router — rule-based scoring engine, intelligent router, and bounded LLM fallback."""

from wmclean.router.intelligent_router import IntelligentRouter
from wmclean.router.llm_fallback import build_llm_fallback_decision
from wmclean.router.rules import route_deterministic

__all__ = [
    "IntelligentRouter",
    "route_deterministic",
    "build_llm_fallback_decision",
]
