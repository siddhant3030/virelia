"""LangChain integration — privacy middleware for ``create_agent`` and LCEL.

Requires the [langchain] extra. The module imports without LangChain
installed; accessing ``VireliaMiddleware`` lazily builds the class (PEP 562)
and raises a clear ImportError if LangChain is missing.

Trust boundary: agent state stays local and keeps real values. Only the
model call is guarded — a masked *copy* of the request goes to the provider,
and tokens in the reply (text and tool-call arguments) are restored before
the reply lands in state. Tools therefore receive real values, the user sees
real values, and the next model call re-masks everything; tokenization is
stable per thread, so the model sees consistent tokens across turns.

    from langchain.agents import create_agent
    from virelia.integrations.langchain import VireliaMiddleware

    agent = create_agent(model, tools, middleware=[VireliaMiddleware(vault=Vault())])
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from virelia.engine import Engine
from virelia.guard import (
    Guard,
    VireliaBlocked,
    apply_content,
    dump_args,
    load_args,
    message_view,
)
from virelia.models import GuardrailResult
from virelia.policy import Policy, load_preset

DEFAULT_PRESET = "agent"

_cached: type | None = None


def __getattr__(name: str) -> Any:
    if name == "VireliaMiddleware":
        global _cached
        if _cached is None:
            _cached = _build_middleware_class()
        return _cached
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _missing(err: ImportError) -> ImportError:
    return ImportError(
        "virelia's LangChain integration requires langchain>=1.0 — install with: "
        "pip install 'virelia[langchain]'"
    )


def make_guard(
    policy: Policy | Engine | Guard | str = DEFAULT_PRESET,
    *,
    vault: Any | None = None,
    hmac_salt: bytes | None = None,
) -> Guard:
    if isinstance(policy, Guard):
        return policy
    if isinstance(policy, str):
        policy = load_preset(policy)
    return Guard(policy, vault=vault, hmac_salt=hmac_salt)


# -- BaseMessage <-> view -------------------------------------------------------


def message_to_view(message: Any) -> dict[str, Any]:
    """Scannable projection of a LangChain message: content, structured
    tool-call args, and the provider-raw copies some chat models serialize
    from ``additional_kwargs`` instead — both must be covered or one leaks."""
    view = message_view(
        message.content, [tc["args"] for tc in getattr(message, "tool_calls", None) or []]
    )
    raw = (getattr(message, "additional_kwargs", None) or {}).get("tool_calls") or []
    view["raw_tool_args"] = [load_args((tc.get("function") or {}).get("arguments")) for tc in raw]
    return view


def view_to_message(message: Any, view: dict[str, Any]) -> Any:
    update: dict[str, Any] = {"content": apply_content(message.content, view["content"])}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        update["tool_calls"] = [
            {**tc, "args": args} for tc, args in zip(tool_calls, view["tool_args"], strict=True)
        ]
    raw = (message.additional_kwargs or {}).get("tool_calls")
    if raw:
        calls = []
        for tc, args in zip(raw, view["raw_tool_args"], strict=True):
            fn = tc.get("function") or {}
            calls.append(
                {**tc, "function": {**fn, "arguments": dump_args(fn.get("arguments"), args)}}
            )
        update["additional_kwargs"] = {**message.additional_kwargs, "tool_calls": calls}
    return message.model_copy(update=update)


def mask_messages(
    guard: Guard, messages: list[Any], *, scope: str | None = None
) -> tuple[list[Any], GuardrailResult]:
    result = guard.scan([message_to_view(m) for m in messages], scope=scope)
    if result.blocked:
        return messages, result
    masked = [view_to_message(m, v) for m, v in zip(messages, result.payload, strict=True)]
    return masked, result


def restore_message(guard: Guard, message: Any, *, scope: str | None = None) -> Any:
    return view_to_message(message, guard.restore(message_to_view(message), scope=scope))


def _restore_structured(guard: Guard, value: Any, scope: str | None) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump") and hasattr(type(value), "model_validate"):
        return type(value).model_validate(guard.restore(value.model_dump(), scope=scope))
    return guard.restore(value, scope=scope)


def thread_scope() -> str | None:
    """The current run's ``thread_id``, so each conversation gets its own
    token namespace. ``None`` outside a LangGraph run or without one."""
    try:
        from langgraph.config import get_config

        thread_id = (get_config().get("configurable") or {}).get("thread_id")
    except Exception:
        return None
    return str(thread_id) if thread_id is not None else None


# -- agent middleware -----------------------------------------------------------


def _build_middleware_class() -> type:
    try:
        from langchain.agents.middleware import AgentMiddleware, ModelResponse
        from langchain_core.messages import AIMessage, ToolMessage
    except ImportError as err:
        raise _missing(err) from err

    class VireliaMiddleware(AgentMiddleware):
        """Masks every model call and restores tokens in the reply.

        policy: a preset name (default ``"agent"``), a Policy, an Engine, or
        a ready-made :class:`virelia.Guard`.
        vault: a :class:`virelia.Vault` — enables reversible tokens; without
        one the engine degrades tokenize to mask.
        scope_by_thread: key token namespaces on the run's ``thread_id``.
        scan_tool_args: check tool arguments under the tool's name before the
        tool runs and refuse the call if policy blocks them (for tools that
        must never receive e.g. credentials). Arguments are never rewritten.
        on_block: ``"respond"`` answers with the policy message instead of
        calling the model; ``"raise"`` raises :class:`VireliaBlocked`.
        """

        def __init__(
            self,
            policy: Policy | Engine | Guard | str = DEFAULT_PRESET,
            *,
            vault: Any | None = None,
            hmac_salt: bytes | None = None,
            scope_by_thread: bool = True,
            scan_tool_args: bool = False,
            on_block: Literal["respond", "raise"] = "respond",
        ) -> None:
            super().__init__()
            self.guard = make_guard(policy, vault=vault, hmac_salt=hmac_salt)
            self.scope_by_thread = scope_by_thread
            self.scan_tool_args = scan_tool_args
            self.on_block = on_block

        def _scope(self) -> str | None:
            return thread_scope() if self.scope_by_thread else None

        # -- model boundary -------------------------------------------------

        def _mask_request(self, request: Any, scope: str | None) -> tuple[Any, Any]:
            system = request.system_message
            outgoing = [system, *request.messages] if system is not None else request.messages
            masked, result = mask_messages(self.guard, outgoing, scope=scope)
            if result.blocked:
                return None, self._blocked_response(result)
            if system is not None:
                return request.override(system_message=masked[0], messages=masked[1:]), None
            return request.override(messages=masked), None

        def _blocked_response(self, result: GuardrailResult) -> Any:
            blocked = VireliaBlocked.from_result(result)
            if self.on_block == "raise":
                raise blocked
            return ModelResponse(result=[AIMessage(content=blocked.reason)])

        def _restore_response(self, response: Any, scope: str | None) -> Any:
            if isinstance(response, AIMessage):
                return restore_message(self.guard, response, scope=scope)
            if not isinstance(response, ModelResponse):
                return response  # e.g. an ExtendedModelResponse from inner middleware
            return ModelResponse(
                result=[restore_message(self.guard, m, scope=scope) for m in response.result],
                structured_response=_restore_structured(
                    self.guard, response.structured_response, scope
                ),
            )

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            scope = self._scope()
            masked, blocked = self._mask_request(request, scope)
            if blocked is not None:
                return blocked
            return self._restore_response(handler(masked), scope)

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            scope = self._scope()
            # The engine is sync and its NER tier is CPU-bound.
            masked, blocked = await asyncio.to_thread(self._mask_request, request, scope)
            if blocked is not None:
                return blocked
            response = await handler(masked)
            return await asyncio.to_thread(self._restore_response, response, scope)

        # -- tool boundary (opt-in) -----------------------------------------

        def _check_tool(self, request: Any, scope: str | None) -> tuple[Any, Any]:
            call = request.tool_call
            result = self.guard.scan(
                call["args"], source=call["name"], direction="request", scope=scope
            )
            if result.blocked:
                blocked = VireliaBlocked.from_result(result)
                if self.on_block == "raise":
                    raise blocked
                return None, ToolMessage(
                    content=blocked.reason,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
            # Refuse-or-pass only: the tool is a trusted sink, so a passing
            # call keeps its real values rather than the masked copy.
            return request, None

        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            if not self.scan_tool_args:
                return handler(request)
            checked, refused = self._check_tool(request, self._scope())
            return refused if refused is not None else handler(checked)

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            if not self.scan_tool_args:
                return await handler(request)
            checked, refused = await asyncio.to_thread(self._check_tool, request, self._scope())
            return refused if refused is not None else await handler(checked)

    return VireliaMiddleware


# -- LCEL ---------------------------------------------------------------------


def masking_runnable(
    policy: Policy | Engine | Guard | str = DEFAULT_PRESET,
    *,
    vault: Any | None = None,
    scope: str | None = None,
) -> Any:
    """A Runnable that masks chain input bound for a model:
    ``masking_runnable(guard) | llm | restoring_runnable(guard)``.

    Accepts a string, a message, a list of messages, a PromptValue, or a
    dict (scanned as data). Raises :class:`VireliaBlocked` when withheld.
    """
    try:
        from langchain_core.messages import BaseMessage
        from langchain_core.prompt_values import PromptValue
        from langchain_core.runnables import RunnableLambda
    except ImportError as err:
        raise _missing(err) from err
    guard = make_guard(policy, vault=vault)

    def mask(value: Any) -> Any:
        if isinstance(value, PromptValue):
            value = value.to_messages()
        if isinstance(value, BaseMessage):
            return mask([value])[0]
        if isinstance(value, list) and all(isinstance(v, BaseMessage) for v in value):
            masked, result = mask_messages(guard, value, scope=scope)
        else:
            result = guard.scan(value, scope=scope)
            masked = result.payload
        if result.blocked:
            raise VireliaBlocked.from_result(result)
        return masked

    return RunnableLambda(mask, name="virelia_mask")


def restoring_runnable(
    policy: Policy | Engine | Guard | str = DEFAULT_PRESET,
    *,
    vault: Any | None = None,
    scope: str | None = None,
) -> Any:
    """A Runnable that restores vault tokens in model output (a message,
    string or data). Pair it with :func:`masking_runnable` sharing one Guard."""
    try:
        from langchain_core.messages import BaseMessage
        from langchain_core.runnables import RunnableLambda
    except ImportError as err:
        raise _missing(err) from err
    guard = make_guard(policy, vault=vault)

    def restore(value: Any) -> Any:
        if isinstance(value, BaseMessage):
            return restore_message(guard, value, scope=scope)
        return guard.restore(value, scope=scope)

    return RunnableLambda(restore, name="virelia_restore")
