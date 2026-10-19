"""Agent-framework integrations, each behind its own extra.

Every integration is a thin mapping from a framework's message and tool
types onto :class:`virelia.Guard` — the guard holds all policy, vault and
audit behavior, so integrations stay small and behave identically.

- ``virelia.integrations.langchain`` — ``VireliaMiddleware`` for
  ``create_agent``, plus LCEL ``masking_runnable`` / ``restoring_runnable``.
"""
