"""Regenerates kaggle/LipVision_Kaggle.ipynb and kaggle/kaggle_cell.py from CELL below.

The notebook is deliberately code-only (no markdown cells): if a notebook is pasted
or imported as plain text, markdown would be executed as Python and fail.
"""

import json
from pathlib import Path

CELL = r'''# 👄 LipVision: real-time silent lip reading on a free Kaggle GPU
#
#  1. Notebook settings (right sidebar > Session options):
#       Accelerator = GPU T4 x2      Internet = On
#  2. Click "Run All". When the output says "LipVision is live", click the button
#     (or use the embedded view), press "Start camera", and mouth sentences silently.
#     Video only: the microphone is never used.
#
#  Weights (~1 GB) download automatically. If Google Drive refuses (quota limit),
#  attach vsr_trlrs2lrs3vox2avsp_base.pth as a Kaggle dataset ("Add Input"); it is
#  found automatically. Private repo? Add a Kaggle secret named GITHUB_TOKEN.

import os, sys, subprocess

REPO = "https://github.com/saidattathreya32/majorproject"
BRANCH = ""  # empty = the repository's default branch
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath(".")
APP = os.path.join(WORK, "majorproject")

token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        out = (r.stdout + r.stderr)[-3000:]
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{out}".replace(token or "\0", "***"))
    return r.stdout


url = REPO.replace("https://", f"https://{token}@") if token else REPO
if os.path.isdir(os.path.join(APP, ".git")):
    sh(["git", "-C", APP, "pull", "-q", "--ff-only"])
else:
    sh(["git", "clone", "-q", "--depth", "1", *(["-b", BRANCH] if BRANCH else []), url, APP])
print("Code:", sh(["git", "-C", APP, "log", "-1", "--format=%h %s"]).strip())

print("Installing packages…", flush=True)
sh([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(APP, "requirements-kaggle.txt")])

os.environ["LIPVISION_HOME"] = os.path.join(WORK, "lipvision_cache")
if APP not in sys.path:
    sys.path.insert(0, APP)  # os.chdir alone does not make a package importable in Jupyter
os.chdir(APP)
for name in [m for m in sys.modules if m == "lipvision" or m.startswith("lipvision.")]:
    del sys.modules[name]  # re-running the cell picks up freshly pulled code

from lipvision.launch import launch_in_notebook

launch_in_notebook()
'''


def main():
    out_dir = Path(__file__).resolve().parent.parent / "kaggle"
    out_dir.mkdir(exist_ok=True)
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": CELL.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (out_dir / "LipVision_Kaggle.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (out_dir / "kaggle_cell.py").write_text(CELL)
    print("wrote", out_dir / "LipVision_Kaggle.ipynb", "and", out_dir / "kaggle_cell.py")


if __name__ == "__main__":
    main()
