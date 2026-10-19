"""virelia: privacy guardrail middleware for AI agents — prompts, tool calls
and MCP traffic, masked before they reach a model provider."""

from virelia.audit import AuditSink, InMemorySink
from virelia.engine import Engine
from virelia.guard import Guard, VireliaBlocked
from virelia.models import Action, AuditEvent, GuardrailResult, Span
from virelia.policy import Policy, PolicyError, load_policy, load_preset, parse_policy
from virelia.vault import Vault, VaultError

__version__ = "0.3.0"

__all__ = [
    "Action",
    "AuditEvent",
    "AuditSink",
    "Engine",
    "Guard",
    "GuardrailResult",
    "InMemorySink",
    "Policy",
    "PolicyError",
    "Span",
    "Vault",
    "VaultError",
    "VireliaBlocked",
    "__version__",
    "load_policy",
    "load_preset",
    "parse_policy",
]
