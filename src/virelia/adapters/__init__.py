"""MCP transport adapters. (Agent-framework integrations such as LangChain
live in ``virelia.integrations`` and build on :class:`virelia.Guard`.)

The engine is transport-agnostic on purpose; an adapter only has to:

1. resolve the policy/engine for the session,
2. call ``engine.scan_request(tool_name, arguments)`` before invoking the
   tool, replacing the arguments with ``result.payload`` (or refusing the
   call when ``result.blocked``),
3. call ``engine.scan_result(tool_name, raw_result)`` on the way back and
   return ``result.payload`` to the client.

That contract keeps each adapter at roughly 150 lines. Planned adapters:

- FastMCP 3.x middleware (``on_call_tool`` hook) for servers you control.
  Note: the FastMCP 1.x bundled in the official ``mcp`` SDK
  (``mcp.server.fastmcp``) has no middleware system — servers on it (e.g.
  dalgo-mcp today) need either a migration or a tool-wrapper shim.
- Standalone proxy that mirrors upstream servers' tools 1:1 for servers
  you can't modify.
"""
