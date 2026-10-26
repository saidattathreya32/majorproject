"""Regenerates kaggle/LipVision_Kaggle.ipynb (keep the notebook in sync with this file)."""

import json
from pathlib import Path

REPO = "https://github.com/saidattathreya32/majorproject"

cells = [
    ("markdown", """# 👄 LipVision — real-time silent lip reading on a free Kaggle GPU

Your laptop only runs the browser (camera + face tracking). The 250M-parameter
lip-reading model runs here on Kaggle's free GPU, and you get a private **https link**
to open on your laptop.

**One-time notebook settings (right sidebar → *Session options*):**
1. **Accelerator → GPU T4 x2** (P100 also works on older PyTorch images; T4 recommended)
2. **Internet → On** (phone verification is required once by Kaggle)

Then **Run All**. The last cell prints a link like `https://xxxx.trycloudflare.com`:
open it in Chrome/Edge, click **Start camera**, and mouth sentences silently.
No microphone is ever used.

> Weights: the notebook downloads the official Auto-AVSR checkpoint automatically. If Google
> Drive refuses (quota), download `vsr_trlrs2lrs3vox2avsp_base.pth` yourself from the
> [Auto-AVSR model zoo](https://github.com/mpc001/auto_avsr#model-zoo), upload it as a Kaggle
> dataset and attach it with **Add Input** — it is picked up from `/kaggle/input` automatically."""),
    ("code", f"""# 1) Get the code (for a private repo add a Kaggle secret named GITHUB_TOKEN)
import os, subprocess
REPO = "{REPO}"
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
url = REPO.replace("https://", f"https://{{token}}@") if token else REPO
os.chdir("/kaggle/working")
if os.path.isdir("majorproject/.git"):
    subprocess.run(["git", "-C", "majorproject", "pull", "-q"], check=True)
else:
    subprocess.run(["git", "clone", "-q", "--depth", "1", url, "majorproject"], check=True)
os.chdir("/kaggle/working/majorproject")
os.environ["LIPVISION_HOME"] = "/kaggle/working/lipvision_cache"
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)"""),
    ("code", """# 2) Install the few extra packages (PyTorch is already on Kaggle)
!pip install -q -r requirements.txt 2>&1 | grep -v "already satisfied" | tail -3
!apt-get -qq install -y libegl1 libgles2 > /dev/null 2>&1 || true   # for MediaPipe in video-file mode
import torch
assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU T4 x2 in the notebook settings"
cap = torch.cuda.get_device_capability()
print(torch.__version__, torch.cuda.get_device_name(), "sm_%d%d" % cap)
if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
    print("⚠️ This PyTorch build does not support this GPU. Switch the accelerator to GPU T4 x2.")"""),
    ("code", """# 3) Download model code + weights (~1 GB, cached for this session) and run a GPU speed test
!python -m lipvision.assets
import numpy as np, time
from lipvision.config import Settings
from lipvision.model import VSRModel
model = VSRModel(Settings.from_env()); model.warmup()
clip = np.random.randint(0, 255, (75, 96, 96), np.uint8)   # 3 s of video
for final in (False, True):
    r = model.transcribe(clip, final=final)
    print(f"{'final (beam search)' if final else 'partial (greedy)':22s} {r.latency_ms:6.0f} ms for a 3 s clip")
del model; torch.cuda.empty_cache()"""),
    ("markdown", """### Optional: transcribe a video file
Upload a silent video (or any video; audio is ignored) as a dataset, then:
```
!python -m lipvision.transcribe /kaggle/input/<your-dataset>/clip.mp4 --save-crops crops.mp4
```
`crops.mp4` shows exactly what the model sees (useful to check alignment)."""),
    ("code", """# 4) Launch the live server + public https link (keep this cell running)
!python scripts/launch_kaggle.py"""),
]


def to_cell(kind, src):
    lines = src.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


nb = {
    "cells": [to_cell(k, s) for k, s in cells],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).resolve().parent.parent / "kaggle" / "LipVision_Kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
