"""Registry and Hub I/O for datasets/<name>/."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "0")

import yaml
from datasets import Dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATASETS = ROOT / "datasets"
_MARKER = ".hf_complete"


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def enable_download_progress() -> None:
    from datasets.utils import enable_progress_bars as enable_dataset_progress_bars
    from huggingface_hub.utils import enable_progress_bars as enable_hub_progress_bars

    enable_dataset_progress_bars()
    enable_hub_progress_bars()


def list_datasets(*, uploadable: bool | None = None) -> list[str]:
    if not DATASETS.is_dir():
        return []
    names = sorted(
        path.name for path in DATASETS.iterdir() if (path / "metadata.yaml").is_file()
    )
    if uploadable is None:
        return names
    return [name for name in names if _metadata(name).get("local_root")]


def _hub(name: str, revision: str | None) -> tuple[str, object, str]:
    meta = _metadata(name)
    schema = _schema(name)
    rev = revision or meta.get("revision") or "main"
    return meta["hf_repo"], schema, rev


def load_hub_dataset(name: str, *, revision: str | None = None, cache_dir: Path | None = None) -> Dataset:
    enable_download_progress()
    repo, schema, rev = _hub(name, revision)
    return schema.load_from_hub(repo, revision=rev, cache_dir=cache_dir)


def load_hub_samples(
    name: str,
    *,
    count: int,
    seed: int | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[dict], list[int], int]:
    enable_download_progress()
    repo, schema, rev = _hub(name, revision)
    if not hasattr(schema, "load_samples_from_hub"):
        raise ValueError(f"Dataset {name!r} does not support sample loading")
    return schema.load_samples_from_hub(
        repo,
        count=count,
        seed=seed,
        revision=rev,
        cache_dir=cache_dir,
    )


def upload(
    name: str,
    *,
    revision: str | None = None,
    private: bool | None = None,
    upload_readme: bool = True,
) -> int:
    meta = _metadata(name)
    schema = _schema(name)
    if not hasattr(schema, "build_from_local"):
        raise ValueError(f"Dataset {name!r} is not uploadable")
    local_root = _local_root(meta, name)
    rev = revision or meta.get("revision") or "main"
    is_private = meta.get("private", True) if private is None else private

    dataset: Dataset = schema.build_from_local(local_root)
    dataset.push_to_hub(meta["hf_repo"], revision=rev, private=is_private)
    print(f"Uploaded {len(dataset)} rows → {meta['hf_repo']} ({rev})")

    if upload_readme:
        readme = DATASETS / name / "README.md"
        HfApi().upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=meta["hf_repo"],
            repo_type="dataset",
            revision=rev,
        )
        print(f"Uploaded README → {meta['hf_repo']}/README.md")
    return len(dataset)


def download(
    name: str,
    *,
    out: Path | None = None,
    revision: str | None = None,
    force: bool = False,
    cache_dir: Path | None = None,
) -> Path:
    meta = _metadata(name)
    schema = _schema(name)
    if not hasattr(schema, "write_local_folders"):
        raise ValueError(f"Dataset {name!r} does not support folder download")
    local_root = _local_root(meta, name)
    target = Path(out) if out is not None else local_root
    rev = revision or meta.get("revision") or "main"

    marker = target / _MARKER
    if not force and marker.is_file():
        print(f"Already downloaded → {target}")
        return target

    enable_download_progress()
    dataset = schema.load_from_hub(meta["hf_repo"], revision=rev, cache_dir=cache_dir)
    written = schema.write_local_folders(dataset, target)
    marker.write_text(f"{meta['hf_repo']}\n{rev}\n", encoding="utf-8")
    print(f"Downloaded {written} rows → {target}")
    return target


def _metadata(name: str) -> dict:
    path = DATASETS / name / "metadata.yaml"
    if not path.is_file():
        known = ", ".join(list_datasets()) or "(none)"
        raise KeyError(f"Unknown dataset {name!r}. Registered: {known}")
    meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not str(meta.get("hf_repo") or "").strip():
        raise ValueError(f"metadata.yaml for {name!r} must set hf_repo")
    if not (DATASETS / name / "README.md").is_file():
        raise FileNotFoundError(f"Missing README.md for dataset {name!r}")
    return meta


def _local_root(meta: dict, name: str) -> Path:
    raw = meta.get("local_root")
    if not raw:
        raise ValueError(f"metadata.yaml for {name!r} must set local_root")
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _schema(name: str):
    path = DATASETS / name / "schema.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing schema.py for dataset {name!r}")
    spec = importlib.util.spec_from_file_location(f"dataset_{name.replace('-', '_')}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load schema module for {name!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "FEATURES") or not hasattr(module, "load_from_hub"):
        raise AttributeError(f"schema.py for {name!r} must define FEATURES and load_from_hub")
    return module
