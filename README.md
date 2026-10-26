# 👄 LipVision 4.0: Real-Time Silent Lip Reading

LipVision reads sentences from lip movement alone, live from your webcam. It uses video only: no microphone is opened and no audio is used.

The heavy model runs on a **free Kaggle GPU**. Your laptop only runs a browser tab that tracks your face and streams a small mouth crop, so it stays cool and responsive.

```
 Laptop (browser)                                   Kaggle T4 GPU (free)
 ┌──────────────────────────────┐   WebSocket     ┌──────────────────────────────────────┐
 │ webcam 640x480 @30fps        │   ~8 KB/frame   │ align mouth to Auto-AVSR mean face   │
 │ MediaPipe face mesh (WebGL)  │ ──────────────▶ │ 96x96 patch, resample to 25 fps      │
 │ lip overlay, activity meter  │                 │ lip-motion speech detector           │
 │ 192x192 lower-face JPEG crop │ ◀────────────── │ Conformer VSR (250M, fp16)           │
 │ live captions + transcript   │  partial/final  │  partial: CTC greedy  (few ms)       │
 └──────────────────────────────┘      text       │  final:   CTC+attention beam search  │
                                                  └──────────────────────────────────────┘
                       free https link via Cloudflare quick tunnel (no account needed)
```

## Quick start (Kaggle, free, recommended)

1. On Kaggle, create a **New Notebook**.
2. In the right sidebar (**Session options**), set **Accelerator → GPU T4 x2** and **Internet → On**.
3. Get the code into the notebook, in either of two ways:
   - **File → Import Notebook**, then upload `kaggle/LipVision_Kaggle.ipynb`, **or**
   - copy the whole of [`kaggle/kaggle_cell.py`](kaggle/kaggle_cell.py) into the first cell.
4. Click **Run All**. After 1–3 minutes, the output shows **"LipVision is live"** with an **Open LipVision** button and an embedded live view.
5. Click **Start camera**, then mouth a sentence silently. Pause for about half a second at the end, and the text appears.

The notebook is a single self-contained cell, so it cannot hit the earlier `SyntaxError` (markdown run as code) or `ModuleNotFoundError` (package not on the import path). If the embedded view can't use your camera (some browsers block cameras inside notebooks), use the **Open** button, which opens a new tab. The app stays live while the notebook session runs. To restart it, run the cell again.

The first run downloads about 1 GB of model weights. If Google Drive refuses the download (popular files sometimes hit a quota limit), get `vsr_trlrs2lrs3vox2avsp_base.pth` from the [Auto-AVSR model zoo](https://github.com/mpc001/auto_avsr#model-zoo), upload it as a Kaggle dataset and attach it with **Add Input**. It is found under `/kaggle/input` automatically.

The notebook works even though this repository is private: it carries a built-in copy of the app and uses it when GitHub asks for a login. To have it always pull the newest code from GitHub instead, add a Kaggle secret named `GITHUB_TOKEN` (**Add-ons → Secrets**) holding a GitHub token with read access, or make the repository public. After changing anything in `lipvision/`, run `python scripts/build_notebook.py` to refresh the built-in copy; a test fails if you forget.

## Run on your own computer (VS Code)

This needs Python 3.10–3.12 and a webcam. An NVIDIA GPU makes it real time; on a laptop CPU it still works, but each final transcript takes a few seconds.

1. Open the `majorproject` folder in VS Code and install the **Python** extension when prompted.
2. **Terminal → Run Task… → "LipVision: Setup"**. This creates `.venv`, installs the requirements and downloads the model (about 1 GB, once). Select `.venv` as the interpreter if VS Code asks.
3. In **Run and Debug** (Ctrl+Shift+D), choose **"LipVision: Live in browser"** and press **F5**. Your browser opens at `http://localhost:8000` when the model is ready; click **Start camera**.

The other run configurations are **Live in OpenCV window** (no browser; there is also a push-to-talk variant using Space), **Transcribe a video file**, and **Run tests**.

From a plain terminal instead:
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # for an NVIDIA GPU, install CUDA torch first (pytorch.org)
python -m lipvision                                      # opens the browser app
python -m lipvision window [--manual]                    # OpenCV window instead
python -m lipvision transcribe clip.mp4 --save-crops crops.mp4
```

### Using the app

| Control | What it does |
|---|---|
| **Auto** mode | Detects speech from lip motion, then shows a live partial transcript followed by the final one |
| **Push-to-talk** | Hold <kbd>Space</kbd> (or the button) while speaking. This is the most accurate mode because the clip boundaries are exact |
| Model view | The 96×96 aligned mouth patch the network sees. If it is not centered on your lips, fix the lighting or your camera angle |
| Lip activity | Motion energy against the adaptive threshold (dashed line) |
| Copy / Save .txt / Read aloud | Export the transcript or speak it with the browser's text-to-speech |

**For the best accuracy:**
- Face the camera with even, frontal light. Avoid a bright window behind you.
- Articulate naturally, at normal speed, and speak in full sentences. The model uses sentence context; isolated single words are the hardest case.
- Keep your mouth visible and unobstructed.
- Fill most of the frame with your face (about arm's length from the camera).

## What changed from the old project (LipRead-3.0)

The old repository (`saidattathreya32/LipVision`) is left untouched. This repository is a clean rebuild. The problems found and how each one is fixed:

| Problem in the old code | Effect | Fix in 4.0 |
|---|---|---|
| Three incompatible pipelines mixed together: a 13-word landmark config, a 3D-CNN pixel classifier (`model.py`), and a non-existent HF model `tiiuae/visper` (`predict_live.py`) | Most scripts could not run together; the accuracy was undefined | One pipeline built on **Auto-AVSR**, an open-vocabulary English sentence model with about 20% WER on LRS3 |
| 13-word vocabulary, trained on about 5 samples per word | Poor accuracy, and only those 13 words | Pretrained on 3,291 hours of video; recognizes any English sentence |
| Raw dlib box crops (`predict_live_3.py`) instead of the alignment the model was trained with | The model saw a distribution it was never trained on, so accuracy dropped | The exact Auto-AVSR preprocessing: a similarity transform to the mean face, a 96×96 mouth patch and landmark smoothing (see `lipvision/preprocess.py`) |
| Open/closed mouth-ratio trigger (`mar > 0.055`) | Cut off utterances on closed-lip sounds (m, b, p) and fired on a still, open mouth | A lip-motion-energy detector with an adaptive noise floor and hangover time, plus push-to-talk |
| Frames assumed to be 25 fps; clips capped at 3 s; inference on the laptop CPU in the UI loop | Speed and time warping, truncated sentences, lag | Timestamp-based 25 fps resampling, clips up to 10 s, inference on a cloud GPU in a separate worker |
| No flow control; frames processed on the UI thread | A growing backlog, which is the "lag" | At most 8 unacknowledged frames plus a socket buffer limit, so frames are never queued stale |
| MediaPipe `mp.solutions` API | Removed in current MediaPipe, so it crashes on install | MediaPipe Tasks API (Python) and tasks-vision (browser) |
| Duplicated and dead files: empty `features.py`/`labels.py`, `container.txt`, `work_environment.txt`, `jason.vscode/`, an empty `auto_avsr/` folder, LFS pointer stubs, a 56 MB `lbfmodel.yaml`, `haarcascade`, `shape_predictor_68` | A heavy, confusing repo | All removed. Weights and upstream code are fetched at runtime and pinned to a specific commit; the repo is about 200 KB |

## Other ways to run

- **Colab or any Linux GPU VM:** run `python -m lipvision cloud`. It starts the server and prints a free public https link (a Cloudflare quick tunnel, with localhost.run as a fallback). Set `LIPVISION_NO_TUNNEL=1` if the machine is directly reachable.
- **Transcribe video files** (offline, using the training-identical centered smoothing): `python -m lipvision transcribe a.mp4 b.mp4`. Add `--save-crops crops.mp4` to see exactly what the model sees.

Browsers allow the camera only on `https://` or `localhost`, which is why remote use goes through the https tunnel.

## Configuration

Every setting in `lipvision/config.py` can be overridden with an environment variable named `LIPVISION_<NAME>`:

| Variable | Default | Meaning |
|---|---|---|
| `LIPVISION_BEAM_SIZE` | auto | Final decode beam width: 10 on a GPU, 4 on a CPU. A larger beam is slightly more accurate and slower |
| `LIPVISION_CHECKPOINT` | `vsr_trlrs2lrs3vox2avsp_base.pth` | Checkpoint name or path (`vsr_trlrs3vox2_base.pth` is a smaller-data alternative) |
| `LIPVISION_CHECKPOINT_URL` | — | Extra download URL to try if Google Drive fails |
| `LIPVISION_VAD_HANGOVER` | `0.6` | Seconds of still lips that end an utterance (lower = faster, but may split sentences) |
| `LIPVISION_PARTIAL_INTERVAL` | auto | Seconds between live partial transcripts: 0.6 on a GPU, 1.5 on a CPU |
| `LIPVISION_MAX_UTTERANCE` | `10` | Longer speech is split and decoded in chunks |
| `LIPVISION_DEVICE` / `LIPVISION_FP16` | `auto` / `true` | Device and half precision |
| `LIPVISION_HOME` | `./.cache` | Where the code, weights and models are cached |

## Project layout

```
lipvision/
  __main__.py     `python -m lipvision` entry point (local / window / cloud / transcribe / download)
  launch.py       cloud launcher: background server + https tunnel + notebook live panel
  config.py       settings, landmark index tables, checkpoint registry
  assets.py       fetches pinned Auto-AVSR code, weights (Kaggle input / Drive / URL), face model
  model.py        VSR model wrapper: fp16 encoder, CTC-greedy partials, joint beam-search finals
  preprocess.py   mean-face alignment, 96x96 mouth patches, smoothing, fps resampling
  vad.py          lip-motion speech activity detector
  stream.py       live session: segmentation, pre-roll, partial/final scheduling
  server.py       FastAPI + WebSocket server, single GPU worker, flow control
  landmarks.py    MediaPipe Tasks face landmarker (video files and local webcam)
  transcribe.py   offline video transcription CLI
  local_live.py   local OpenCV webcam app
  web/            browser client (index.html, style.css, app.js)
kaggle/LipVision_Kaggle.ipynb   one-click Kaggle notebook (single code cell)
kaggle/kaggle_cell.py           the same cell as plain text, for copy-paste
scripts/build_notebook.py       regenerates both files above
.vscode/                        run configurations, setup task, settings
tests/                          pytest suite (runs without weights, using a fake model)
```

Run the tests with `pip install -r requirements-dev.txt && pytest`. The requirement files are `requirements.txt` for local use and `requirements-kaggle.txt` for the server extras on Kaggle.

## Limitations

- Lip reading is hard even for humans. The benchmark WER of about 20% on LRS3 is measured on TED-style video; webcam conditions, accents and lighting can make the error rate higher.
- English only. Many words look identical on the lips (for example "pat", "bat" and "mat"), and sentence context is what resolves them.
- The Cloudflare quick-tunnel URL changes every time the notebook restarts. Kaggle sessions stop after inactivity and have a weekly GPU quota (currently about 30 hours).

## Credits and licenses

- Model, tokenizer and mean face: [Auto-AVSR](https://github.com/mpc001/auto_avsr) by Ma et al. (ICASSP 2023), Apache-2.0. These are downloaded at runtime and not redistributed here. Check the checkpoint terms in the upstream repository before any commercial use.
- Face tracking: [MediaPipe](https://github.com/google-ai-edge/mediapipe), Apache-2.0.
- Tunnel: [cloudflared](https://github.com/cloudflare/cloudflared), Apache-2.0.

© 2026 Sai Dattathreya. Project code (everything outside the third-party components above).
