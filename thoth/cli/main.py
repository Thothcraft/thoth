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
from ..settings import ConfigStore


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
def daemon(window, tick):
    """Run the persistent node service in the foreground."""
    from ..daemon import ThothDaemon
    click.echo("Starting Thoth daemon (Ctrl+C to stop)…")
    ThothDaemon(window_seconds=window, tick_hz=tick).run_forever()


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
