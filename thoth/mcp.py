"""MCP (Model Context Protocol) stdio server for Thoth.

Exposes the platform to MCP hosts (Claude Desktop, agents, IDEs) as
tools backed by the same context/event model as the REST API — no
separate logic: context reads ``/v1/context``, events stream over
``/v1/events/stream`` (SSE), collection/actuation ride the node relay.

Run: ``thoth mcp`` — speaks JSON-RPC 2.0 on stdin/stdout (newline-
delimited). Credentials come from ``WHISPY_API_KEY`` or
``~/.whispy/credentials.json`` (``whispy login``).
"""

from __future__ import annotations

import json
import sys
import threading
import queue as _queue
from typing import Any, Callable, Dict, Optional

PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {"name": "thoth", "version": "1.0.0"}


def _client():
    from whispy.cloud.client import Client
    return Client()


TOOLS = [
    {
        "name": "thoth_devices",
        "description": "List the user's Thoth nodes: id, name, online, "
                       "health summary.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "thoth_context",
        "description": "Current context states (predictions, occupancy, "
                       "any estimator output). Optional key/entity filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "entity_id": {"type": "string"},
                "snapshot": {"type": "boolean",
                            "description": "return full context snapshot "
                                           "(entities+relationships+states)"},
            },
        },
    },
    {
        "name": "thoth_events",
        "description": "Recent node events (predictions, trigger_fired, "
                       "notification, room_changed). Filter by kind/device.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "kind": {"type": "string"},
                "since": {"type": "string",
                          "description": "event id or epoch ts"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "thoth_wait_event",
        "description": "Subscribe to the live event stream and return the "
                       "first matching event (no polling). Blocks up to "
                       "timeout_s.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "kind": {"type": "string"},
                "timeout_s": {"type": "number", "default": 30},
            },
        },
    },
    {
        "name": "thoth_capture_start",
        "description": "Start a synchronized sensor capture on a node "
                       "(all sensors by default — native rates preserved).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "sensors": {"type": "array", "items": {"type": "string"}},
                "label": {"type": "string",
                          "description": "manual label for the session"},
            },
            "required": ["device_id"],
        },
    },
    {
        "name": "thoth_capture_stop",
        "description": "Stop a running capture by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"capture_id": {"type": "string"}},
            "required": ["capture_id"],
        },
    },
    {
        "name": "thoth_node_api",
        "description": "Relay an arbitrary call to a node's local API over "
                       "the Brain WS tunnel (e.g. GET /api/v1/health).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "method": {"type": "string", "default": "GET"},
                "path": {"type": "string"},
                "body": {"type": "object"},
            },
            "required": ["device_id", "path"],
        },
    },
    {
        "name": "thoth_add_rule",
        "description": "Create a server-side automation rule "
                       "{when: {key, entity_id?, equals?, min_confidence?}, "
                       "then: {device_id, actuator_id, operation, params}} — "
                       "edge-triggered, fires on context transitions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "when": {"type": "object"},
                "then": {"type": "object"},
                "cooldown_s": {"type": "number", "default": 0},
            },
            "required": ["name", "when", "then"],
        },
    },
    {
        "name": "thoth_subscribe_webhook",
        "description": "POST every matching node event to a URL "
                       "(HMAC-signed, retried). Returns the secret once.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "kinds": {"type": "array", "items": {"type": "string"}},
                "device_id": {"type": "string"},
            },
            "required": ["url"],
        },
    },
]


def _call_tool(name: str, args: Dict[str, Any]) -> Any:
    c = _client()
    if name == "thoth_devices":
        return {"devices": [d.info.to_dict() for d in c.devices()]}
    if name == "thoth_context":
        if args.get("snapshot"):
            return c.context_snapshot()
        return {"states": c.context(key=args.get("key"),
                                    entity_id=args.get("entity_id"),
                                    active_only=True)}
    if name == "thoth_events":
        return {"events": c.events(device_id=args.get("device_id"),
                                   kind=args.get("kind"),
                                   since=args.get("since"),
                                   limit=int(args.get("limit") or 50))}
    if name == "thoth_wait_event":
        timeout = float(args.get("timeout_s") or 30)
        q: "_queue.Queue[Dict[str, Any]]" = _queue.Queue(maxsize=1)

        def _reader() -> None:
            try:
                for evt in c.event_stream(device_id=args.get("device_id"),
                                          kind=args.get("kind")):
                    q.put(evt)
                    return
            except Exception as exc:
                q.put({"error": str(exc)})

        threading.Thread(target=_reader, daemon=True).start()
        try:
            return q.get(timeout=timeout)
        except _queue.Empty:
            return {"timeout": timeout, "received": False}
    if name == "thoth_capture_start":
        return c.capture_start(device_id=args["device_id"],
                               sensors=args.get("sensors"),
                               label=args.get("label"))
    if name == "thoth_capture_stop":
        return c.capture_stop(args["capture_id"])
    if name == "thoth_node_api":
        return c.node_api(device_id=args["device_id"],
                          method=args.get("method", "GET"),
                          path=args["path"], body=args.get("body"))
    if name == "thoth_add_rule":
        return c.add_rule(name=args["name"], when=args["when"],
                          then=args["then"],
                          cooldown_s=float(args.get("cooldown_s") or 0))
    if name == "thoth_subscribe_webhook":
        return c.subscribe_webhook(url=args["url"], kinds=args.get("kinds"),
                                   device_id=args.get("device_id"))
    raise KeyError(f"unknown tool {name!r}")


def _result(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def _handle(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        return _result(rid, {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": SERVER_INFO,
            "capabilities": {"tools": {}}})
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return _result(rid, {})
    if method == "tools/list":
        return _result(rid, {"tools": TOOLS})
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            out = _call_tool(name, args)
            return _result(rid, {
                "content": [{"type": "text",
                             "text": json.dumps(out, indent=2)}],
                "isError": False})
        except Exception as exc:
            return _result(rid, {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True})
    if method == "resources/list":
        return _result(rid, {"resources": []})
    if method == "prompts/list":
        return _result(rid, {"prompts": []})
    if rid is None:
        return None  # notification — no reply
    return _error(rid, -32601, f"method {method!r} not implemented")


def serve() -> None:
    """Newline-delimited JSON-RPC 2.0 on stdio."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(
                _error(None, -32700, "parse error")) + "\n")
            sys.stdout.flush()
            continue
        resp = _handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


__all__ = ["serve", "TOOLS"]
