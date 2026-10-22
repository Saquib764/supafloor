"""Load HF_TOKEN from this package or a sibling trainer .env."""

from __future__ import annotations

import os
from pathlib import Path

def _package_dirs() -> list[Path]:
    found: list[Path] = []
    here = Path.cwd()
    for _ in range(8):
        pyproject = here / "pyproject.toml"
        if pyproject.is_file() and 'name = "gemma-author"' in pyproject.read_text(encoding="utf-8"):
            found.append(here)
            break
        if here.parent == here:
            break
        here = here.parent
    found.append(Path(__file__).resolve().parent)
    return found


def _dotenv_paths() -> list[Path]:
    paths: list[Path] = []
    for root in _package_dirs():
        paths.append(root / ".env")
        backend = root.parent
        for name in (
            "decision_gemma",
            "modern_bert_image_gen",
            "florence_trainer",
            "sana_trainer",
        ):
            paths.append(backend / name / ".env")
    return paths


def _apply_dotenv(env_path: Path, *, override: bool) -> None:
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def load_project_dotenv() -> None:
    """Sibling trainer .env files fill gaps. This package's .env wins."""
    package_roots = {path.resolve() for path in _package_dirs()}
    package_envs: list[Path] = []
    sibling_envs: list[Path] = []
    for env_path in _dotenv_paths():
        if not env_path.is_file():
            continue
        if env_path.name == ".env" and env_path.parent.resolve() in package_roots:
            package_envs.append(env_path)
        else:
            sibling_envs.append(env_path)
    for env_path in sibling_envs:
        _apply_dotenv(env_path, override=False)
    for env_path in package_envs:
        _apply_dotenv(env_path, override=True)


def hf_hub_token() -> str | None:
    load_project_dotenv()
    token = ""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            break
    if not token:
        return None
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token
