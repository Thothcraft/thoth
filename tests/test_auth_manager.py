import base64
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend.auth_manager import AuthManager


def _token(expiry: int) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"sub": "7", "exp": expiry}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


class _Config:
    def __init__(self, root: Path, token: str):
        self.CONFIG_DIR = str(root / "config")
        self.LEGACY_CONFIG_DIR = str(root / "legacy")
        self.DATA_DIR = str(root)
        self.BRAIN_AUTH_TOKEN = token
        self.USER_AUTH_TOKEN = token


def test_expired_config_token_is_removed_from_runtime(tmp_path):
    config = _Config(tmp_path, _token(int(time.time()) - 60))

    manager = AuthManager(config)

    assert manager.is_authenticated() is False
    assert config.BRAIN_AUTH_TOKEN == ""
    assert config.USER_AUTH_TOKEN == ""


def test_expired_saved_token_is_deleted_and_not_retried(tmp_path):
    token = _token(int(time.time()) - 60)
    config = _Config(tmp_path, "")
    auth_file = Path(config.CONFIG_DIR) / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text(json.dumps({"token": token}), encoding="utf-8")

    manager = AuthManager(config)

    assert manager.token is None
    assert not auth_file.exists()
    assert config.BRAIN_AUTH_TOKEN == ""


def test_valid_config_token_is_available_to_device_scheduler(tmp_path):
    token = _token(int(time.time()) + 3600)
    config = _Config(tmp_path, token)

    manager = AuthManager(config)

    assert manager.is_authenticated() is True
    assert config.USER_AUTH_TOKEN == token
    assert config.BRAIN_AUTH_TOKEN == token
