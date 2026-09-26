"""Registry and Hub I/O for datasets/<name>/."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import yaml
from datasets import Dataset
from huggingface_hub import HfApi

def _manager_root() -> Path:
    """Resolve dataset_manager/ even when registry.py is copied into site-packages."""
    for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        manager = parent if parent.name == "dataset_manager" else parent / "dataset_manager"
        if (manager / "registry.py").is_file() and (manager / "datasets").is_dir():
            return manager
    raise FileNotFoundError("dataset_manager/datasets not found")


ROOT = _manager_root()
REPO_ROOT = ROOT.parent
DATASETS = ROOT / "datasets"
_MARKER = ".hf_complete"


def _load_dotenv_from(env_dir: Path) -> None:
    env_path = env_dir / ".env"
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


def load_dotenv(*env_dirs: Path) -> None:
    for env_dir in env_dirs or (ROOT, Path.cwd()):
        _load_dotenv_from(env_dir)


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


def local_dataset_root(name: str) -> Path | None:
    meta = _metadata(name)
    if not meta.get("local_root"):
        return None
    return _local_root(meta, name)


def local_dataset_available(name: str) -> bool:
    root = local_dataset_root(name)
    if root is None or not root.is_dir():
        return False
    if (root / "input").is_dir():
        return True
    schema = _schema(name)
    if hasattr(schema, "local_available"):
        return schema.local_available(root)
    return False


def load_local_dataset(name: str) -> Dataset:
    meta = _metadata(name)
    schema = _schema(name)
    if not hasattr(schema, "build_from_local"):
        raise ValueError(f"Dataset {name!r} does not support local loading")
    return schema.build_from_local(_local_root(meta, name))


def load_dataset(
    name: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    source: str = "auto",
) -> Dataset:
    """Load from local_root when available (or forced), otherwise from the Hub."""
    if source not in ("auto", "local", "hub"):
        raise ValueError(f"source must be auto, local, or hub, got {source!r}")
    if source in ("auto", "local") and local_dataset_available(name):
        schema = _schema(name)
        if hasattr(schema, "build_from_local"):
            dataset = load_local_dataset(name)
            if len(dataset) > 0:
                return dataset
            if source == "local":
                raise FileNotFoundError(f"No rows under local dataset {name!r}")
    if source == "local":
        raise FileNotFoundError(f"Local folder not found for dataset {name!r}")
    return load_hub_dataset(name, revision=revision, cache_dir=cache_dir)


def dataset_row_count(
    name: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    source: str = "auto",
) -> int:
    return len(load_dataset(name, revision=revision, cache_dir=cache_dir, source=source))


def load_rows_at(
    dataset: Dataset,
    indices: range | list[int],
    *,
    columns: tuple[str, ...] | None = None,
) -> list[dict]:
    rows = dataset
    if columns is not None:
        missing = [col for col in columns if col not in rows.column_names]
        if missing:
            raise KeyError(f"Dataset missing columns: {missing}")
        rows = rows.select_columns(list(columns))
    return [rows[int(index)] for index in indices]


def hub_dataset_row_count(
    name: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> int:
    from datasets import DownloadMode, load_dataset

    enable_download_progress()
    repo, _, rev = _hub(name, revision)
    kwargs: dict[str, object] = {
        "split": "train",
        "revision": rev,
        "download_mode": DownloadMode.REUSE_DATASET_IF_EXISTS,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return len(load_dataset(repo, **kwargs))


def load_hub_rows_at(
    name: str,
    indices: range | list[int],
    *,
    columns: tuple[str, ...] | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> list[dict]:
    from datasets import DownloadMode, load_dataset

    enable_download_progress()
    repo, _, rev = _hub(name, revision)
    kwargs: dict[str, object] = {
        "split": "train",
        "revision": rev,
        "download_mode": DownloadMode.REUSE_DATASET_IF_EXISTS,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    dataset = load_dataset(repo, **kwargs)
    if columns is not None:
        missing = [col for col in columns if col not in dataset.column_names]
        if missing:
            raise KeyError(f"Hub dataset missing columns: {missing}")
        dataset = dataset.select_columns(list(columns))
    return [dataset[index] for index in indices]


def load_hub_stream(
    name: str,
    *,
    split: str = "train",
    columns: tuple[str, ...] | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
):
    from datasets import DownloadMode, load_dataset

    enable_download_progress()
    repo, _, rev = _hub(name, revision)
    kwargs: dict[str, object] = {
        "split": split,
        "revision": rev,
        "streaming": True,
        "download_mode": DownloadMode.REUSE_DATASET_IF_EXISTS,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    dataset = load_dataset(repo, **kwargs)
    if columns is not None:
        missing = [col for col in columns if col not in dataset.column_names]
        if missing:
            raise KeyError(f"Hub dataset missing columns: {missing}")
        dataset = dataset.select_columns(list(columns))
    return dataset


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
    if not hasattr(schema, "materialize_local") and not hasattr(schema, "write_local_folders"):
        raise ValueError(f"Dataset {name!r} does not support folder download")
    local_root = _local_root(meta, name)
    target = Path(out) if out is not None else local_root
    rev = revision if revision is not None else meta.get("revision")
    if rev is not None and str(rev).lower() in ("null", "none", ""):
        rev = None

    marker = target / _MARKER
    if not force and marker.is_file():
        print(f"Already downloaded → {target}")
        return target

    enable_download_progress()
    subset = str(meta.get("race_subset") or "all")
    if hasattr(schema, "materialize_local"):
        print(f"Downloading {name!r} → {target} (subset={subset})", flush=True)
        written = schema.materialize_local(
            target,
            repo=meta["hf_repo"],
            subset=subset,
            cache_dir=cache_dir,
            revision=rev,
        )
    elif hasattr(schema, "write_local_folders"):
        dataset = schema.load_from_hub(meta["hf_repo"], revision=rev, cache_dir=cache_dir)
        written = schema.write_local_folders(dataset, target)
    else:
        raise ValueError(f"Dataset {name!r} does not support folder download")
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
