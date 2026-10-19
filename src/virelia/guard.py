"""Framework-neutral guard: the piece every agent integration builds on.

An agent has exactly one untrusted party — the model provider. A Guard masks
what crosses toward it (messages, prompts) and restores what comes back
(tokens in replies and tool-call arguments), so tools and end users keep
working with real values while the provider only ever sees stand-ins.

Restoring only does something when the policy tokenizes and a vault is
configured; partial masks, masks and redactions are one-way by design.

    guard = Guard(load_preset("agent"), vault=Vault())
    masked = guard.mask_messages(messages)        # → send masked.payload
    reply = guard.restore_messages([reply_msg])   # → real values back

    @guard.tool                                   # any Python callable
    def send_sms(phone: str, body: str) -> dict: ...
"""

from __future__ import annotations

import functools
import inspect
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

from virelia.audit import AuditSink
from virelia.engine import Engine
from virelia.models import AuditEvent, GuardrailResult
from virelia.policy import Policy

LLM_SOURCE = "llm"

F = TypeVar("F", bound=Callable[..., Any])


class VireliaBlocked(Exception):
    """A payload was withheld by policy. Never carries the raw value."""

    def __init__(
        self,
        reason: str,
        *,
        rule_id: str | None = None,
        events: list[AuditEvent] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.rule_id = rule_id
        self.events = events or []

    @classmethod
    def from_result(cls, result: GuardrailResult) -> VireliaBlocked:
        rule = result.payload.get("rule") if isinstance(result.payload, dict) else None
        return cls(
            result.block_reason or "blocked by virelia", rule_id=rule, events=result.events
        )


class Guard:
    """Masks payloads bound for the model; restores tokens coming back.

    ``vault`` (a :class:`virelia.Vault`) enables reversible tokenization. Its
    file is shared across scopes; pass ``scope=`` to any method to keep
    conversations apart — the same value gets independent tokens per scope.
    """

    def __init__(
        self,
        policy: Policy | Engine,
        *,
        vault: Any | None = None,
        sink: AuditSink | None = None,
        hmac_salt: bytes | None = None,
    ) -> None:
        if isinstance(policy, Engine):
            self.engine = policy
            vault = vault or policy.vault
        else:
            self.engine = Engine(policy, sink=sink, hmac_salt=hmac_salt)
        self._vault = vault
        self._scoped: dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- vault scoping ------------------------------------------------------

    def vault_for(self, scope: str | None = None) -> Any | None:
        """The vault bound to ``scope``; ``None`` scope is the default vault."""
        if self._vault is None or scope is None or scope == getattr(self._vault, "scope", None):
            return self._vault
        with self._lock:
            if scope not in self._scoped:
                self._scoped[scope] = type(self._vault)(Path(self._vault.path), scope=scope)
            return self._scoped[scope]

    def close(self) -> None:
        with self._lock:
            for vault in self._scoped.values():
                vault.close()
            self._scoped.clear()

    # -- masking (toward the model) -----------------------------------------

    def scan(
        self,
        payload: Any,
        *,
        source: str = LLM_SOURCE,
        direction: Literal["request", "result"] = "request",
        scope: str | None = None,
    ) -> GuardrailResult:
        return self.engine.scan(
            payload, source=source, direction=direction, vault=self.vault_for(scope)
        )

    def mask_text(
        self, text: str, *, source: str = LLM_SOURCE, scope: str | None = None
    ) -> GuardrailResult:
        return self.scan(text, source=source, scope=scope)

    def mask_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        source: str = LLM_SOURCE,
        scope: str | None = None,
    ) -> GuardrailResult:
        """Mask OpenAI-style chat messages. Only text is scanned — content
        strings, ``{"type": "text"}`` blocks and tool-call arguments — never
        roles, ids or non-text blocks. ``payload`` holds new message dicts."""
        views = [message_view(m.get("content"), _dict_tool_args(m)) for m in messages]
        result = self.scan(views, source=source, scope=scope)
        if result.blocked:
            return result
        result.payload = [
            _apply_dict_view(m, v) for m, v in zip(messages, result.payload, strict=True)
        ]
        return result

    # -- restoring (from the model) -----------------------------------------

    def restore(self, payload: Any, *, scope: str | None = None) -> Any:
        """Swap vault tokens back for originals. Identity without a vault."""
        vault = self.vault_for(scope)
        return payload if vault is None else vault.rehydrate(payload)

    def restore_messages(
        self, messages: list[dict[str, Any]], *, scope: str | None = None
    ) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            view = message_view(m.get("content"), _dict_tool_args(m))
            out.append(_apply_dict_view(m, self.restore(view, scope=scope)))
        return out

    # -- tools ----------------------------------------------------------------

    @overload
    def tool(self, fn: F) -> F: ...

    @overload
    def tool(
        self, *, name: str | None = None, scan_args: bool = False, scope: str | None = None
    ) -> Callable[[F], F]: ...

    def tool(
        self,
        fn: F | None = None,
        *,
        name: str | None = None,
        scan_args: bool = False,
        scope: str | None = None,
    ) -> F | Callable[[F], F]:
        """Guard any callable used as an agent tool.

        Tokens in arguments are restored (the tool is a trusted sink), then
        — if ``scan_args`` — the arguments are checked under the tool's name
        and the call is refused if policy blocks them (they are never
        rewritten), and the result is masked before it can travel back to
        the model.
        Raises :class:`VireliaBlocked` when either side is withheld.
        Works for sync and async callables and keeps the signature, so
        framework decorators like LangChain's ``@tool`` can stack on top.
        """

        def decorate(func: F) -> F:
            tool_name = name or func.__name__
            sig = inspect.signature(func)

            def before(args: tuple, kwargs: dict) -> inspect.BoundArguments:
                bound = sig.bind(*args, **kwargs)
                arguments = self.restore(dict(bound.arguments), scope=scope)
                if scan_args:
                    # Refuse-or-pass only: rewriting would re-tokenize the
                    # values just restored, and the tool is the trusted sink.
                    checked = self.scan(
                        arguments, source=tool_name, direction="request", scope=scope
                    )
                    if checked.blocked:
                        raise VireliaBlocked.from_result(checked)
                bound.arguments.update(arguments)
                return bound

            def after(result: Any) -> Any:
                checked = self.scan(result, source=tool_name, direction="result", scope=scope)
                if checked.blocked:
                    raise VireliaBlocked.from_result(checked)
                return checked.payload

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    bound = before(args, kwargs)
                    return after(await func(*bound.args, **bound.kwargs))

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                bound = before(args, kwargs)
                return after(func(*bound.args, **bound.kwargs))

            return wrapper  # type: ignore[return-value]

        return decorate(fn) if fn is not None else decorate


# -- message views ------------------------------------------------------------
#
# A view is the scannable projection of one message: {"content": ...,
# "tool_args": [...]}. Content keeps its shape (a string, or a list aligned
# with the original blocks where non-text blocks become None, which the
# engine skips), so scan paths read like $[3].content[1]. Integrations build
# views from their own message types and write the scanned view back.


def message_view(content: Any, tool_args: list[Any] | None = None) -> dict[str, Any]:
    if isinstance(content, list):
        content = [_block_text(b) for b in content]
    elif not isinstance(content, str):
        content = None
    return {"content": content, "tool_args": tool_args or []}


def apply_content(original: Any, scanned: Any) -> Any:
    """Write a scanned view's content back into the original content shape."""
    if not isinstance(original, list):
        return scanned if isinstance(original, str) else original
    out = []
    for block, text in zip(original, scanned, strict=True):
        if isinstance(block, str):
            out.append(text)
        elif _block_text(block) is not None:
            out.append({**block, "text": text})
        else:
            out.append(block)
    return out


def _block_text(block: Any) -> str | None:
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and block.get("type") == "text":
        text = block.get("text")
        return text if isinstance(text, str) else None
    return None


def load_args(raw: Any) -> Any:
    """Tool-call arguments as data: JSON strings are parsed so paths and
    restored values stay structural (re-serialization escapes them)."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def dump_args(original: Any, value: Any) -> Any:
    if isinstance(original, str) and not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return value


def _dict_tool_args(message: dict[str, Any]) -> list[Any]:
    return [
        load_args((tc.get("function") or {}).get("arguments"))
        for tc in message.get("tool_calls") or []
    ]


def _apply_dict_view(message: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    new = dict(message)
    if "content" in message:
        new["content"] = apply_content(message["content"], view["content"])
    if message.get("tool_calls"):
        calls = []
        for tc, args in zip(message["tool_calls"], view["tool_args"], strict=True):
            fn = tc.get("function") or {}
            calls.append(
                {**tc, "function": {**fn, "arguments": dump_args(fn.get("arguments"), args)}}
            )
        new["tool_calls"] = calls
    return new
