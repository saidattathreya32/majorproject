"""Regenerates kaggle/LipVision_Kaggle.ipynb and kaggle/kaggle_cell.py.

The notebook is a single code cell (markdown cells get executed as Python when a
notebook is pasted or imported as text). It also embeds a zipped copy of the
`lipvision` package, so Run All works even when the GitHub repo is private and
no token is configured; when the repo is reachable, git is used to stay current.

Run this after changing anything under lipvision/ (a test checks it is in sync).
"""

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLED = ["lipvision", "requirements-kaggle.txt"]

HEAD = r'''# 👄 LipVision: real-time silent lip reading on a free Kaggle GPU
#
#  1. Notebook settings (right sidebar > Session options):
#       Accelerator = GPU T4 x2      Internet = On
#  2. Click "Run All". When the output says "LipVision is live", click the button
#     (or use the embedded view), press "Start camera", and mouth sentences silently.
#     Video only: the microphone is never used.
#
#  Works with a private repo too: if GitHub can't be reached without a login, the copy
#  of LipVision built into this cell is used. (Optional: add a Kaggle secret named
#  GITHUB_TOKEN to always pull the latest code from GitHub.)
#  Weights (~1 GB) download automatically. If Google Drive refuses (quota limit),
#  attach vsr_trlrs2lrs3vox2avsp_base.pth as a Kaggle dataset ("Add Input").

import base64, io, os, shutil, subprocess, sys, zipfile

# MODE = "live"      -> open the live lip-reading app (default)
# MODE = "evaluate"  -> measure accuracy on your clips: attach a dataset that holds the
#                       videos and a manifest.csv (path, speaker, reference), then Run All.
#                       Results are saved under /kaggle/working/results/<EVAL_LABEL>.
MODE = "live"
EVAL_LABEL = "baseline"

REPO = "https://github.com/saidattathreya32/majorproject"
BRANCH = ""  # empty = the repository's default branch
PIN = ""     # e.g. "v1.0" to always run that release; empty = newest working release
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath(".")
APP = os.path.join(WORK, "majorproject")


def github_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return None


def git(*args, token=None):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # never hang on a login prompt
    r = subprocess.run(["git", *args], capture_output=True, text=True, env=env)
    msg = (r.stdout + r.stderr).strip().replace(token or "\0", "***")
    return r.returncode == 0, msg


def release_tags(url, token):
    """Release tags (v1.0, v1.1, ...) on the remote, newest first. Empty if unreachable."""
    ok, out = git("ls-remote", "--tags", url, token=token)
    tags = {line.split("refs/tags/")[1].removesuffix("^{}") for line in out.splitlines()
            if ok and "refs/tags/v" in line}

    def key(t):
        try:
            return [int(x) for x in t[1:].split(".")]
        except ValueError:
            return [-1]
    return sorted((t for t in tags if key(t) != [-1]), key=key, reverse=True)


def candidates():
    """Versions to try, best first: pinned or newest release, older releases, then fallbacks."""
    token = github_token()
    url = REPO.replace("https://", f"https://{token}@") if token else REPO
    tags = release_tags(url, token)
    if PIN:
        tags = [PIN] + [t for t in tags if t != PIN]
    out = [(t, url, token) for t in tags[:3]]  # newest 3 releases is enough history
    if not tags:
        out.append((BRANCH or "latest", url, token))
    out.append(("built-in", None, None))
    return out


def get_code(ref, url, token):
    """Put version `ref` in APP. Returns a description, or raises if it cannot be fetched."""
    os.chdir(WORK)  # a failed attempt left us inside APP; deleting it would break git and pip
    shutil.rmtree(APP, ignore_errors=True)
    if ref == "built-in":
        os.makedirs(APP)
        zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(BUNDLE)))).extractall(APP)
        return "built-in copy (known-good snapshot inside this cell)"
    branch = [] if ref == "latest" else ["-b", ref]
    ok, msg = git("clone", "-q", "--depth", "1", *branch, url, APP, token=token)
    if not ok:
        raise RuntimeError("GitHub not reachable without a login: " + msg[-200:])
    return f"GitHub {ref}"


def start(ref, url, token):
    print(f"Code source: {get_code(ref, url, token)}", flush=True)
    print("Installing packages…", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        os.path.join(APP, "requirements-kaggle.txt")], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("pip install failed (is Internet on?):\n" + r.stderr[-2000:])
    os.environ["LIPVISION_HOME"] = os.path.join(WORK, "lipvision_cache")
    if APP not in sys.path:
        sys.path.insert(0, APP)  # os.chdir alone does not make a package importable in Jupyter
    os.chdir(APP)
    for name in [m for m in sys.modules if m == "lipvision" or m.startswith("lipvision.")]:
        del sys.modules[name]  # re-running the cell (or falling back) picks up fresh code
    if MODE == "evaluate":
        from lipvision.evaluate import run_in_notebook
        return run_in_notebook(EVAL_LABEL)
    from lipvision.launch import launch_in_notebook
    return launch_in_notebook()


def main():
    """Try the newest release; if it fails to start, fall back to the previous one automatically."""
    failures = []
    for ref, url, token in candidates():
        try:
            result = start(ref, url, token)
        except Exception as e:
            failures.append(f"{ref}: {e}")
            print(f"⚠️ {ref} failed: {str(e)[-400:]}\n   Falling back to the previous version…", flush=True)
            try:
                from lipvision.launch import stop
                stop()  # free the port before the next attempt
            except Exception:
                pass
            continue
        print(f"✅ Running {ref}" + (f" (fallback after: {'; '.join(failures)})" if failures else ""))
        return result
    raise RuntimeError("Every version failed to start:\n" + "\n".join(failures))


# Zipped copy of the lipvision package (generated by scripts/build_notebook.py).
'''

TAIL = "\n\nmain()\n"


def bundle_bytes() -> bytes:
    """Deterministic zip of the app (fixed timestamps, sorted entries)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for top in BUNDLED:
            p = ROOT / top
            files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".pyc":
                    continue
                info = zipfile.ZipInfo(f.relative_to(ROOT).as_posix(), date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, f.read_bytes())
    return buf.getvalue()


def build_cell() -> str:
    b64 = base64.b64encode(bundle_bytes()).decode()
    lines = ",\n".join(f'    "{b64[i:i + 100]}"' for i in range(0, len(b64), 100))
    return HEAD + f"BUNDLE = [\n{lines},\n]" + TAIL


def main():
    cell = build_cell()
    out_dir = ROOT / "kaggle"
    out_dir.mkdir(exist_ok=True)
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": cell.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (out_dir / "LipVision_Kaggle.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (out_dir / "kaggle_cell.py").write_text(cell)
    print("wrote", out_dir / "LipVision_Kaggle.ipynb", "and", out_dir / "kaggle_cell.py",
          f"({len(cell) // 1024} KB)")


if __name__ == "__main__":
    main()
