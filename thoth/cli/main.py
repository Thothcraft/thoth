"""``thoth`` CLI — controls the daemon via local IPC.

    thoth daemon              run the node service in the foreground
    thoth status              daemon + device status
    thoth sensors             sensor inventory
    thoth capture start|stop|list
    thoth models              installed runtime models
    thoth predict <label>     inject a prediction (drives actuators)
    thoth pair                start device pairing with Brain
    thoth doctor              diagnostics
"""

from __future__ import annotations

import json
import sys

import click

from ..ipc import DaemonClient, DaemonUnavailable
from ..settings import ConfigStore, config_dir


# -- daemon lifecycle (pidfile) -----------------------------------------------
def _pidfile():
    return config_dir() / "daemon.pid"


def _read_pid() -> int | None:
    try:
        return int(json.loads(_pidfile().read_text())["pid"])
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        import os
        os.kill(pid, 0)
        return True
    except (OSError, OverflowError):
        return False


def _write_pid() -> None:
    import os
    _pidfile().write_text(json.dumps({"pid": os.getpid(),
                                     "started_at": __import__("time").time()}))


def _clear_pid() -> None:
    try:
        _pidfile().unlink(missing_ok=True)
    except OSError:
        pass


def _client(ctx) -> DaemonClient:
    return ctx.obj["client"]


def _echo(data) -> None:
    click.echo(json.dumps(data, indent=2, default=str))


@click.group()
@click.option("--port", default=None, type=int, help="daemon local API port")
@click.pass_context
def main(ctx, port):
    """Thoth node CLI."""
    ctx.ensure_object(dict)
    cfg = ConfigStore()
    ctx.obj["config"] = cfg
    ctx.obj["client"] = DaemonClient(cfg, port=port)


@main.command()
@click.option("--window", default=2.0, type=float, help="window seconds")
@click.option("--tick", default=2.0, type=float, help="loop rate Hz")
@click.option("--stop", is_flag=True, help="stop the running daemon")
@click.option("--status", "status_", is_flag=True, help="show daemon state")
@click.option("--dashboard/--no-dashboard", "dashboard", default=None,
              help="serve the local dashboard UI (default: on, or the "
                   "dashboard_enabled config key)")
@click.option("--dashboard-port", "dashboard_port", default=None, type=int,
              help="port for the dashboard UI (default: 80, or the "
                   "dashboard_port config key; falls back to the API port "
                   "when unbindable)")
@click.pass_context
def daemon(ctx, window, tick, stop, status_, dashboard, dashboard_port):
    """Run the node service in the foreground (single instance).

    ``thoth daemon --stop`` / ``--status`` control a running daemon via
    its pidfile (~/.thoth/daemon.pid); a second start refuses while one
    is already running.
    """
    import os
    import signal

    if status_:
        pid = _read_pid()
        if pid and _pid_alive(pid):
            click.echo(f"daemon running: pid {pid}")
        else:
            _clear_pid()
            click.echo("daemon not running")
            sys.exit(1)
        return

    if stop:
        pid = _read_pid()
        if not pid or not _pid_alive(pid):
            _clear_pid()
            click.echo("daemon not running", err=True)
            sys.exit(1)
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            if not _pid_alive(pid):
                break
            __import__("time").sleep(0.3)
        _clear_pid()
        click.echo(f"daemon stopped (pid {pid})")
        return

    existing = _read_pid()
    if existing and _pid_alive(existing):
        click.echo(f"daemon already running (pid {existing}) — refusing "
                   "to start a second instance", err=True)
        sys.exit(1)
    _clear_pid()

    from ..daemon import ThothDaemon
    _write_pid()
    d = ThothDaemon(window_seconds=window, tick_hz=tick,
                    serve_ui=dashboard, dashboard_port=dashboard_port)
    # SIGTERM → graceful stop so --stop works cross-platform.
    signal.signal(signal.SIGTERM, lambda *_: d.stop() or sys.exit(0))
    click.echo("Starting Thoth daemon (Ctrl+C to stop)…")
    try:
        d.run_forever()
    finally:
        _clear_pid()


@main.command()
@click.pass_context
def status(ctx):
    """Show daemon + device status."""
    try:
        _echo(_client(ctx).status())
    except DaemonUnavailable as exc:
        click.echo(f"daemon not running: {exc}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def sensors(ctx):
    """List the node's sensor inventory."""
    try:
        _echo(_client(ctx).get("/api/sensors"))
    except DaemonUnavailable as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def actuators(ctx):
    """List the node's actuator inventory."""
    try:
        _echo(_client(ctx).get("/api/actuators"))
    except DaemonUnavailable as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


@main.command()
@click.option("--lan", is_flag=True, help="bind the local API to the LAN")
@click.option("--off", is_flag=True, help="return to loopback-only")
@click.option("--sensor", "sensors_", multiple=True,
              help="expose only these sensor ids (repeatable)")
@click.option("--actuator", "actuators_", multiple=True,
              help="expose only these actuator ids (repeatable)")
@click.option("--port", "expose_port", default=None, type=int)
@click.pass_context
def expose(ctx, lan, off, sensors_, actuators_, expose_port):
    """Configure LAN exposure of this node's capabilities.

    The daemon binds loopback by default; ``thoth expose --lan`` opts
    into LAN access. ``--sensor``/``--actuator`` restrict which
    capabilities are visible — permissions describe capabilities, not
    HTTP routes. Restart the daemon to apply.
    """
    cfg = ctx.obj["config"]
    if off:
        cfg.set("local_host", "127.0.0.1")
        click.echo("LAN exposure disabled — daemon will bind 127.0.0.1 "
                   "on next start.")
        return
    if not lan and not (sensors_ or actuators_ or expose_port):
        raise click.UsageError("specify --lan, --off, or capability options")
    if lan:
        cfg.set("local_host", "0.0.0.0")
    if expose_port:
        cfg.set("local_port", int(expose_port))
    if sensors_ or actuators_:
        exp = dict(cfg.get("exposed") or {})
        if sensors_:
            exp["sensors"] = list(sensors_)
        if actuators_:
            exp["actuators"] = list(actuators_)
        cfg.set("exposed", exp)
    host = cfg.get("local_host", "127.0.0.1")
    port = cfg.get("local_port", 5000)
    click.echo(f"local API will bind {host}:{port} on next daemon start")
    exp = cfg.get("exposed") or {}
    if exp.get("sensors") or exp.get("actuators"):
        click.echo(f"exposed sensors:   {exp.get('sensors') or 'all'}")
        click.echo(f"exposed actuators: {exp.get('actuators') or 'all'}")
    else:
        click.echo("exposed capabilities: all")
    click.echo(f"local token: {cfg.local_token}")


@main.group()
def capture():
    """Manage captures."""


@capture.command("start")
@click.argument("sensors", nargs=-1)
@click.pass_context
def capture_start(ctx, sensors):
    """Start a capture over the named sensors (or all)."""
    try:
        _echo(_client(ctx).post("/api/captures/start",
                                {"sensors": list(sensors)}))
    except DaemonUnavailable as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


@capture.command("stop")
@click.argument("capture_id")
@click.pass_context
def capture_stop(ctx, capture_id):
    """Stop a running capture."""
    try:
        _echo(_client(ctx).post("/api/captures/stop",
                                {"capture_id": capture_id}))
    except DaemonUnavailable as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


@capture.command("list")
@click.pass_context
def capture_list(ctx):
    """List captures."""
    try:
        _echo(_client(ctx).get("/api/captures"))
    except DaemonUnavailable as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def models(ctx):
    """List installed runtime models."""
    try:
        _echo(_client(ctx).get("/api/models"))
    except DaemonUnavailable as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


@main.command()
@click.argument("label")
@click.option("--confidence", default=1.0, type=float)
@click.pass_context
def predict(ctx, label, confidence):
    """Inject a prediction — drives linked actuators."""
    try:
        _echo(_client(ctx).post("/api/internal/prediction",
                                {"class": label, "confidence": confidence}))
    except DaemonUnavailable as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


def _brain_req(method: str, url: str, body=None,
               token=None, secret=None):
    """Minimal JSON request — stdlib only, no new deps."""
    import urllib.request
    import urllib.error
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    if secret:
        r.add_header("X-Pairing-Secret", secret)
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}


@main.command()
@click.option("--user", "user", default=None,
              help="portal username/email (prompted when omitted)")
@click.option("--password", default=None,
              help="portal password (prompted when omitted)")
@click.option("--brain", "brain", default=None,
              help="Brain base URL (default: configured or "
                   "https://api.thothcraft.com)")
@click.pass_context
def pair(ctx, user, password, brain):
    """Pair this node with the portal (Brain) — full handshake.

    Logs in as the portal user, starts device pairing, claims the code,
    and writes the resulting device token into ~/.thoth/config.json.
    A running daemon picks it up on its next heartbeat (~30 s); otherwise
    restart with ``thoth daemon --stop`` + ``thoth daemon``.
    """
    import socket
    import time as _time

    cfg = ctx.obj["config"]
    brain = (brain or cfg.brain_url).rstrip("/")

    if not user:
        user = click.prompt("Portal username or email")
    if not password:
        password = click.prompt("Portal password", hide_input=True)

    # 1. user login → JWT for the claim call
    st, body = _brain_req("POST", f"{brain}/api/token",
                          {"username": user, "password": password})
    if st != 200 or not body.get("access_token"):
        click.echo(f"login failed: {st} {body.get('detail') or body}",
                   err=True)
        sys.exit(1)
    user_token = body["access_token"]

    # 2. start pairing (unauthenticated device call)
    st, body = _brain_req("POST", f"{brain}/api/device/pairing/start", {
        "device_id": cfg.device_id,
        "device_name": cfg.device_name,
        "device_type": "thoth",
        "hardware_info": {"hostname": socket.gethostname()}})
    if st != 200:
        click.echo(f"pairing/start failed: {st} "
                   f"{body.get('detail') or body}", err=True)
        sys.exit(1)
    code, secret = body["code"], body["pairing_secret"]

    # 3. claim the code as the user
    st, body = _brain_req("POST", f"{brain}/api/device/pairing/claim",
                          {"code": code}, token=user_token)
    if st != 200:
        click.echo(f"pairing/claim failed: {st} "
                   f"{body.get('detail') or body}", err=True)
        sys.exit(1)

    # 4. poll status → device JWT
    body = {}
    for _ in range(15):
        st, body = _brain_req("GET", f"{brain}/api/device/pairing/status",
                              secret=secret)
        if body.get("status") == "paired":
            break
        _time.sleep(1)
    if body.get("status") != "paired" or not body.get("access_token"):
        click.echo(f"pairing/status timed out: {body}", err=True)
        sys.exit(1)

    cfg.set("device_token", body["access_token"])
    cfg.set("brain_url", brain)
    cfg.set("paired_user", (body.get("user") or {}).get("username"))
    click.echo(f"paired — {cfg.device_name} ({cfg.device_id}) → {brain}")
    click.echo("The daemon picks up the token on its next heartbeat; "
               "restart it to connect immediately.")


@main.command()
@click.option("--remote", is_flag=True,
              help="query Brain /v1/context instead of the local daemon")
@click.pass_context
def context(ctx, remote):
    """Current context — local daemon or Brain context store."""
    if remote:
        from whispy.cloud.client import Client
        try:
            _echo(Client().context_snapshot())
        except Exception as exc:
            click.echo(f"Brain context failed: {exc}", err=True)
            sys.exit(1)
        return
    try:
        _echo(_client(ctx).get("/api/v1/context"))
    except DaemonUnavailable as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


@main.command()
@click.option("--brain", is_flag=True,
              help="read the Brain event feed instead of local daemon state")
@click.option("--kind", default=None, help="filter by event kind")
@click.option("--since", default=None,
              help="event id or epoch ts — strict >")
@click.option("--device", "device_id", default=None)
@click.option("--follow", is_flag=True,
              help="stream live events over SSE (no polling)")
@click.pass_context
def events(ctx, brain, kind, since, device_id, follow):
    """Event feed — node events via Brain (or local status when offline)."""
    if brain or follow:
        from whispy.cloud.client import Client
        try:
            client = Client()
        except Exception as exc:
            click.echo(f"no Brain credentials ({exc}) — run `whispy login` "
                       f"or set WHISPY_API_KEY", err=True)
            sys.exit(1)
        if not follow:
            _echo(client.events(device_id=device_id, kind=kind,
                                since=since))
            return
        last_id = since
        while True:  # drop-reconnect loop with resume cursor
            try:
                for evt in client.event_stream(
                        device_id=device_id, kind=kind,
                        last_event_id=last_id):
                    _echo(evt)
                    last_id = str(evt.get("id") or last_id or "")
            except KeyboardInterrupt:
                return
            except Exception as exc:
                click.echo(f"stream dropped ({exc}); reconnecting…",
                           err=True)
                import time as _t
                _t.sleep(2.0)
        return
    try:
        _echo(_client(ctx).get("/api/v1/context"))
    except DaemonUnavailable as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)


@main.command()
def mcp():
    """Run the MCP stdio server (agents/IDEs ↔ Thoth tools)."""
    from ..mcp import serve
    serve()


@main.command()
@click.pass_context
def doctor(ctx):
    """Run node diagnostics."""
    from ..diagnostics import run_doctor
    _echo(run_doctor(ctx.obj["config"], _client(ctx)))


if __name__ == "__main__":
    main(obj={})
