"""Fetches everything the engine needs at runtime so the git repo stays small.

* Auto-AVSR model code + tokenizer + mean face (pinned commit, Apache-2.0)
* Pretrained VSR checkpoint (searched locally / in Kaggle inputs, else downloaded)
* MediaPipe face landmarker (only for offline video and local-webcam modes)
"""

import glob
import hashlib
import io
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import (AUTO_AVSR_COMMIT, AUTO_AVSR_REPO, CHECKPOINTS,
                     FACE_LANDMARKER_URL, Settings)

log = logging.getLogger("lipvision.assets")


def auto_avsr_dir(settings: Settings) -> Path:
    """Return the pinned Auto-AVSR checkout, downloading it if missing, and put it on sys.path."""
    target = settings.cache_dir / f"auto_avsr-{AUTO_AVSR_COMMIT[:8]}"
    marker = target / "espnet" / "nets" / "pytorch_backend" / "e2e_asr_conformer.py"
    if not marker.exists():
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("Downloading Auto-AVSR code @ %s", AUTO_AVSR_COMMIT[:8])
        url = f"{AUTO_AVSR_REPO}/archive/{AUTO_AVSR_COMMIT}.zip"
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(tmp)
            inner = next(tmp.iterdir())
            shutil.rmtree(target, ignore_errors=True)
            inner.rename(target)
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:  # zip download blocked -> fall back to git
            log.warning("zip download failed (%s); trying git clone", e)
            shutil.rmtree(target, ignore_errors=True)
            subprocess.check_call(["git", "clone", "-q", AUTO_AVSR_REPO, str(target)])
            subprocess.check_call(["git", "-C", str(target), "checkout", "-q", AUTO_AVSR_COMMIT])
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    return target


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_valid(path: Path, md5_prefix: str | None) -> bool:
    if not path.is_file() or path.stat().st_size < 50_000_000:  # LFS pointer / HTML error page
        return False
    if md5_prefix and not _md5(path).startswith(md5_prefix):
        log.warning("%s: md5 mismatch (expected %s...)", path, md5_prefix)
        return False
    return True


def checkpoint_path(settings: Settings) -> Path:
    """Locate (or download) the VSR checkpoint.

    Search order: LIPVISION_CHECKPOINT as a path -> cache dir -> any Kaggle input
    dataset -> Google Drive (official release) -> LIPVISION_CHECKPOINT_URL.
    """
    name = settings.checkpoint
    explicit = Path(name).expanduser()
    if explicit.suffix == ".pth" and explicit.is_file():
        return explicit
    name = explicit.name
    gdrive_id, md5_prefix, _ = CHECKPOINTS.get(name, (None, None, None))

    candidates = [settings.cache_dir / name]
    candidates += [Path(p) for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True)]
    candidates += [Path(p) for p in glob.glob("/kaggle/input/**/vsr_*.pth", recursive=True)]
    for c in candidates:
        if _looks_valid(c, md5_prefix if c.name == name else None):
            log.info("Using checkpoint %s", c)
            return c

    dest = settings.cache_dir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    if gdrive_id:
        try:
            import gdown
            log.info("Downloading %s from Google Drive", name)
            gdown.download(id=gdrive_id, output=str(dest), quiet=False)
            if _looks_valid(dest, md5_prefix):
                return dest
            errors.append("Google Drive returned an invalid file (quota limit?)")
        except Exception as e:
            errors.append(f"Google Drive: {e}")
    url = os.environ.get("LIPVISION_CHECKPOINT_URL")
    if url:
        try:
            log.info("Downloading %s from %s", name, url)
            urllib.request.urlretrieve(url, dest)
            if _looks_valid(dest, None):
                return dest
            errors.append("LIPVISION_CHECKPOINT_URL returned an invalid file")
        except Exception as e:
            errors.append(f"{url}: {e}")
    raise FileNotFoundError(
        f"Could not obtain {name}. Tried: {'; '.join(errors) or 'nothing'}.\n"
        "Fix: download it from https://github.com/mpc001/auto_avsr#model-zoo and either add it "
        "to the Kaggle notebook as a dataset (Add Input), put it in "
        f"{settings.cache_dir}, or set LIPVISION_CHECKPOINT=/path/to/{name}.")


def face_landmarker_path(settings: Settings) -> Path:
    dest = settings.cache_dir / "face_landmarker.task"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading MediaPipe face landmarker")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, dest)
    return dest


def main():
    """`python -m lipvision.assets` pre-fetches everything (used by the Kaggle notebook)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = Settings.from_env()
    print("auto_avsr :", auto_avsr_dir(s))
    print("landmarker:", face_landmarker_path(s))
    print("checkpoint:", checkpoint_path(s))


if __name__ == "__main__":
    main()
