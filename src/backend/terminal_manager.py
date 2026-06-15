"""Interactive SSH terminal sessions for Thoth dashboards."""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pexpect

logger = logging.getLogger(__name__)

USERNAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
HARDWARE_GROUPS = ("dialout", "video", "spi", "i2c", "gpio", "render", "plugdev")


def _safe_username(username: str) -> str:
    candidate = (username or "").strip()
    if not candidate:
        raise ValueError("Username is required for SSH provisioning")
    if not USERNAME_RE.fullmatch(candidate):
        raise ValueError(
            "Username must use lowercase letters, digits, hyphens, or underscores, and start with a letter"
        )
    return candidate


def _ensure_directory(path: Path, mode: int = 0o700) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except Exception:
        pass


def _resolve_command(*names: str) -> Optional[str]:
    for name in names:
        if os.path.isabs(name) and os.path.exists(name):
            return name
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return None


@dataclass
class TerminalSession:
    sid: str
    username: str
    ssh_user: str
    private_key: Path
    process: pexpect.spawn
    stop_event: threading.Event = field(default_factory=threading.Event)
    reader: Optional[threading.Thread] = None


class SSHTerminalManager:
    """Manage one SSH-backed terminal per browser session."""

    def __init__(self, socketio, config):
        self.socketio = socketio
        self.config = config
        self._lock = threading.RLock()
        self._sessions: Dict[str, TerminalSession] = {}
        self._key_dir = Path(getattr(config, "CONFIG_DIR", Path.home() / ".config" / "thoth")) / "ssh"
        self._key_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_ssh_service(self) -> None:
        if os.name != "posix":
            return
        for action in (["systemctl", "enable", "--now", "ssh"], ["systemctl", "start", "ssh"]):
            try:
                subprocess.run(action, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                continue

    def ensure_user(self, username: str) -> str:
        """Create a local Linux account for the Thoth user if needed."""
        ssh_user = _safe_username(username)

        if os.name != "posix":
            return ssh_user

        home_dir = Path("/home") / ssh_user
        private_key = self._key_dir / f"{ssh_user}_ed25519"
        public_key = private_key.with_suffix(".pub")

        try:
            import pwd

            pwd.getpwnam(ssh_user)
            user_exists = True
        except KeyError:
            user_exists = False

        if not user_exists:
            logger.info("Creating local SSH user: %s", ssh_user)
            useradd_bin = _resolve_command("/usr/sbin/useradd", "useradd")
            adduser_bin = _resolve_command("/usr/sbin/adduser", "adduser")
            if useradd_bin:
                subprocess.run(
                    [useradd_bin, "-m", "-s", "/bin/bash", "-U", ssh_user],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif adduser_bin:
                subprocess.run(
                    [
                        adduser_bin,
                        "--disabled-password",
                        "--gecos",
                        "",
                        "--shell",
                        "/bin/bash",
                        ssh_user,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                raise RuntimeError("No local user creation command found (useradd/adduser)")

        for group in HARDWARE_GROUPS:
            try:
                subprocess.run(
                    ["usermod", "-aG", group, ssh_user],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                continue

        _ensure_directory(home_dir, 0o755)
        ssh_dir = home_dir / ".ssh"
        _ensure_directory(ssh_dir, 0o700)

        if not private_key.exists() or not public_key.exists():
            logger.info("Generating SSH keypair for %s", ssh_user)
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(private_key)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        auth_keys = ssh_dir / "authorized_keys"
        public_key_text = public_key.read_text(encoding="utf-8").strip()
        existing = auth_keys.read_text(encoding="utf-8") if auth_keys.exists() else ""
        if public_key_text not in existing:
            with open(auth_keys, "a", encoding="utf-8") as handle:
                if existing and not existing.endswith("\n"):
                    handle.write("\n")
                handle.write(public_key_text + "\n")

        try:
            import pwd
            import grp

            pw_entry = pwd.getpwnam(ssh_user)
            gid = pw_entry.pw_gid
            os.chown(home_dir, pw_entry.pw_uid, gid)
            os.chown(ssh_dir, pw_entry.pw_uid, gid)
            os.chown(auth_keys, pw_entry.pw_uid, gid)
            os.chown(private_key, 0, 0)
            os.chown(public_key, 0, 0)
            os.chmod(auth_keys, 0o600)
            os.chmod(private_key, 0o600)
            os.chmod(public_key, 0o644)

            for path in (home_dir, ssh_dir, auth_keys):
                try:
                    os.chown(path, pw_entry.pw_uid, gid)
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Unable to finalize SSH permissions for %s: %s", ssh_user, exc)

        self._ensure_ssh_service()
        return ssh_user

    def _spawn_ssh(self, ssh_user: str, private_key: Path) -> pexpect.spawn:
        ssh_bin = shutil.which("ssh")
        if not ssh_bin:
            raise RuntimeError("ssh client is not installed")

        cmd = [
            ssh_bin,
            "-tt",
            "-i",
            str(private_key),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "PreferredAuthentications=publickey",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "PubkeyAuthentication=yes",
            f"{ssh_user}@127.0.0.1",
        ]

        return pexpect.spawn(
            cmd[0],
            cmd[1:],
            encoding="utf-8",
            codec_errors="replace",
            timeout=1,
            dimensions=(34, 132),
        )

    def open(self, sid: str, username: str) -> Dict[str, str]:
        ssh_user = self.ensure_user(username)
        private_key = self._key_dir / f"{ssh_user}_ed25519"

        with self._lock:
            self.close(sid, emit_closed=False)
            process = self._spawn_ssh(ssh_user, private_key)
            session = TerminalSession(
                sid=sid,
                username=username,
                ssh_user=ssh_user,
                private_key=private_key,
                process=process,
            )
            self._sessions[sid] = session
            session.reader = threading.Thread(target=self._reader_loop, args=(session,), daemon=True)
            session.reader.start()

        return {
            "session_id": sid,
            "ssh_user": ssh_user,
            "host": socket.gethostname(),
        }

    def _reader_loop(self, session: TerminalSession) -> None:
        process = session.process
        try:
            while not session.stop_event.is_set():
                try:
                    chunk = process.read_nonblocking(size=4096, timeout=0.2)
                except pexpect.TIMEOUT:
                    continue
                except pexpect.EOF:
                    break
                except Exception as exc:
                    logger.debug("Terminal read error for %s: %s", session.sid, exc)
                    break

                if chunk:
                    self.socketio.emit(
                        "terminal_output",
                        {"session_id": session.sid, "data": chunk},
                        to=session.sid,
                    )
        finally:
            self.close(session.sid)

    def send(self, sid: str, data: str) -> bool:
        with self._lock:
            session = self._sessions.get(sid)
        if not session or session.stop_event.is_set():
            return False

        try:
            if data in ("\u0003", "\x03"):
                session.process.sendcontrol("c")
            elif data in ("\u001a", "\x1a"):
                session.process.sendcontrol("z")
            else:
                session.process.send(data)
            return True
        except Exception as exc:
            logger.debug("Failed to send terminal input for %s: %s", sid, exc)
            return False

    def resize(self, sid: str, rows: int, cols: int) -> None:
        with self._lock:
            session = self._sessions.get(sid)
        if not session or session.stop_event.is_set():
            return

        try:
            session.process.setwinsize(max(10, rows), max(40, cols))
        except Exception as exc:
            logger.debug("Failed to resize terminal for %s: %s", sid, exc)

    def close(self, sid: str, emit_closed: bool = True) -> None:
        with self._lock:
            session = self._sessions.pop(sid, None)

        if not session:
            return

        session.stop_event.set()
        try:
            if session.process.isalive():
                session.process.terminate(force=True)
        except Exception:
            pass

        if emit_closed:
            try:
                self.socketio.emit("terminal_closed", {"session_id": sid}, to=sid)
            except Exception:
                pass
