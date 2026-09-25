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
@click.pass_context
def daemon(ctx, window, tick, stop, status_, dashboard):
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
    d = ThothDaemon(window_seconds=window, tick_hz=tick, serve_ui=dashboard)
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


@main.command()
@click.pass_context
def pair(ctx):
    """Start device pairing with Brain (prints a claim code)."""
    cfg = ctx.obj["config"]
    click.echo(f"device_id: {cfg.device_id}")
    click.echo(f"device_name: {cfg.device_name}")
    click.echo(f"brain: {cfg.brain_url}")
    click.echo("Pairing handshake is driven by the daemon once a Brain "
               "device token is configured (BRAIN_AUTH_TOKEN).")


@main.command()
@click.pass_context
def doctor(ctx):
    """Run node diagnostics."""
    from ..diagnostics import run_doctor
    _echo(run_doctor(ctx.obj["config"], _client(ctx)))


if __name__ == "__main__":
    main(obj={})
