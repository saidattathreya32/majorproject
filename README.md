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

## Quick start (Kaggle, free)

1. Create a new Kaggle notebook: **File → Import Notebook**, then upload `kaggle/LipVision_Kaggle.ipynb` from this repo.
2. In **Session options**, set **Accelerator → GPU T4 x2** and **Internet → On**.
3. **Run All**. The last cell prints a link such as `https://something.trycloudflare.com`.
4. Open the link in Chrome or Edge on your laptop, click **Start camera**, and mouth a sentence silently. Pause for about half a second at the end, and the text appears.

The first run downloads about 1 GB of model weights. If Google Drive refuses the download (popular files sometimes hit a quota limit), get `vsr_trlrs2lrs3vox2avsp_base.pth` from the [Auto-AVSR model zoo](https://github.com/mpc001/auto_avsr#model-zoo), upload it as a Kaggle dataset and attach it with **Add Input**. The notebook finds it under `/kaggle/input` automatically.

If this repository is private, add a Kaggle secret named `GITHUB_TOKEN` (**Add-ons → Secrets**) holding a GitHub token with read access.

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

**On a machine with an NVIDIA GPU (local webcam window):**
```bash
pip install -r requirements.txt
python -m lipvision.local_live            # add --manual for push-to-talk with SPACE
```

**Self-hosted browser mode on any GPU box:**
```bash
python -m lipvision.server --port 8000    # open http://localhost:8000
```
Browsers allow the camera only on `https://` or `localhost`. For remote access, use `scripts/launch_kaggle.py`, which also works on Colab or any Linux VM.

**Transcribe video files (offline, uses the training-identical centered smoothing):**
```bash
python -m lipvision.transcribe clip1.mp4 clip2.mp4 --save-crops crops.mp4
```

## Configuration

Every setting in `lipvision/config.py` can be overridden with an environment variable named `LIPVISION_<NAME>`:

| Variable | Default | Meaning |
|---|---|---|
| `LIPVISION_BEAM_SIZE` | `10` | Final decode beam width. A larger beam is slightly more accurate and slower |
| `LIPVISION_CHECKPOINT` | `vsr_trlrs2lrs3vox2avsp_base.pth` | Checkpoint name or path (`vsr_trlrs3vox2_base.pth` is a smaller-data alternative) |
| `LIPVISION_CHECKPOINT_URL` | — | Extra download URL to try if Google Drive fails |
| `LIPVISION_VAD_HANGOVER` | `0.6` | Seconds of still lips that end an utterance (lower = faster, but may split sentences) |
| `LIPVISION_PARTIAL_INTERVAL` | `0.6` | Seconds between live partial transcripts |
| `LIPVISION_MAX_UTTERANCE` | `10` | Longer speech is split and decoded in chunks |
| `LIPVISION_DEVICE` / `LIPVISION_FP16` | `auto` / `true` | Device and half precision |
| `LIPVISION_HOME` | `./.cache` | Where the code, weights and models are cached |

## Project layout

```
lipvision/
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
kaggle/LipVision_Kaggle.ipynb   one-click Kaggle notebook (generated by scripts/build_notebook.py)
scripts/launch_kaggle.py        server + Cloudflare tunnel launcher
tests/                          pytest suite (runs without weights, using a fake model)
```

Run the tests with `pip install -r requirements-dev.txt && pytest`.

## Limitations

- Lip reading is hard even for humans. The benchmark WER of about 20% on LRS3 is measured on TED-style video; webcam conditions, accents and lighting can make the error rate higher.
- English only. Many words look identical on the lips (for example "pat", "bat" and "mat"), and sentence context is what resolves them.
- The Cloudflare quick-tunnel URL changes every time the notebook restarts. Kaggle sessions stop after inactivity and have a weekly GPU quota (currently about 30 hours).

## Credits and licenses

- Model, tokenizer and mean face: [Auto-AVSR](https://github.com/mpc001/auto_avsr) by Ma et al. (ICASSP 2023), Apache-2.0. These are downloaded at runtime and not redistributed here. Check the checkpoint terms in the upstream repository before any commercial use.
- Face tracking: [MediaPipe](https://github.com/google-ai-edge/mediapipe), Apache-2.0.
- Tunnel: [cloudflared](https://github.com/cloudflare/cloudflared), Apache-2.0.

© 2026 Sai Dattathreya. Project code (everything outside the third-party components above).
