"""Cloud launcher (Kaggle / Colab / any Linux GPU VM): server + public https tunnel.

From a notebook:    from lipvision.launch import launch_in_notebook; launch_in_notebook()
From a terminal:    python -m lipvision cloud

The server runs as a background process, so the notebook cell finishes and the
app stays live while the kernel is running. Re-running restarts it cleanly.
"""

import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .config import Settings

ROOT = Path(__file__).resolve().parent.parent
_procs: list = []


def _log_dir(settings: Settings) -> Path:
    d = Path(settings.cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def stop():
    """Stop a server/tunnel started earlier in this Python process."""
    while _procs:
        p = _procs.pop()
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


def start_server(settings: Settings, port: int) -> subprocess.Popen:
    log = open(_log_dir(settings) / "server.log", "w")
    env = dict(os.environ, PYTHONUNBUFFERED="1", LIPVISION_HOME=str(settings.cache_dir))
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen([sys.executable, "-m", "lipvision.server", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    _procs.append(proc)
    return proc


def server_log(settings: Settings, lines: int = 25) -> str:
    try:
        return "".join((_log_dir(settings) / "server.log").read_text(errors="replace").splitlines(True)[-lines:])
    except OSError:
        return ""


def wait_ready(settings: Settings, port: int, proc: subprocess.Popen, timeout: float = 1800) -> dict:
    """Block until the model is loaded. Raises RuntimeError with the reason on failure."""
    t0, last = time.time(), ""
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("Server exited during start-up:\n" + server_log(settings))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as r:
                info = json.load(r)
            if info["ready"]:
                return info
            if info["error"]:
                raise RuntimeError(info["error"])
        except OSError:
            pass
        tail = server_log(settings, 1).strip()
        if tail and tail != last:  # show download / loading progress
            print("  ", tail[-160:], flush=True)
            last = tail
        time.sleep(3)
    raise RuntimeError("Server did not become ready in time:\n" + server_log(settings))


def _cloudflared(settings: Settings) -> Path:
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    path = _log_dir(settings) / "cloudflared"
    if not path.exists():
        arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
        urllib.request.urlretrieve(url, path)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _tunnel(cmd: list, pattern: str, timeout: float = 60) -> str | None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _procs.append(proc)
    found: list = []

    def read():  # keep draining the log forever so the tunnel never blocks on a full pipe
        for line in proc.stdout:
            m = re.search(pattern, line)
            if m and not found:
                found.append(m.group(0))

    threading.Thread(target=read, daemon=True).start()
    t0 = time.time()
    while not found and time.time() - t0 < timeout and proc.poll() is None:
        time.sleep(0.5)
    return found[0] if found else None


def open_tunnel(settings: Settings, port: int) -> str:
    """Free public https URL: Cloudflare quick tunnel, falling back to localhost.run."""
    try:
        url = _tunnel([str(_cloudflared(settings)), "tunnel", "--no-autoupdate", "--url",
                       f"http://127.0.0.1:{port}"], r"https://[a-z0-9-]+\.trycloudflare\.com")
        if url:
            return url
    except OSError as e:
        print("cloudflared unavailable:", e)
    if shutil.which("ssh"):
        print("Cloudflare tunnel failed, trying localhost.run…")
        url = _tunnel(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
                       "-R", f"80:127.0.0.1:{port}", "nokey@localhost.run"],
                      r"https://[a-z0-9-]+\.lhr\.life")
        if url:
            return url
    raise RuntimeError("Could not open a public tunnel. Is Internet enabled in the notebook settings?")


def gpu_report() -> str:
    try:
        import torch
    except ImportError:
        return "PyTorch not installed"
    if not torch.cuda.is_available():
        return ("⚠️ No GPU visible: in Kaggle open Settings → Accelerator → GPU T4 x2, then Run All again. "
                "(It will still work on CPU, just slower.)")
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0]}{cap[1]}"
    msg = f"GPU: {torch.cuda.get_device_name()} ({arch}), torch {torch.__version__}"
    if torch.cuda.get_arch_list() and arch not in torch.cuda.get_arch_list():
        msg += "\n⚠️ This PyTorch build does not support this GPU. Choose 'GPU T4 x2' as the accelerator."
    return msg


def launch(port: int = 8000, settings: Settings | None = None) -> str:
    settings = settings or Settings.from_env()
    stop()
    print(gpu_report())
    print("Starting LipVision server (first run downloads ~1 GB of weights, 1-3 min)…", flush=True)
    proc = start_server(settings, port)
    try:
        info = wait_ready(settings, port, proc)
        print(f"Model ready on {info['device']} (fp16={info['fp16']}, beam={info['beam']})")
        if os.environ.get("LIPVISION_NO_TUNNEL"):  # direct access (e.g. a VM with an open port)
            return f"http://127.0.0.1:{port}/"
        print("Opening public https tunnel…", flush=True)
        return open_tunnel(settings, port)
    except BaseException:
        stop()
        raise


def launch_in_notebook(port: int = 8000, embed: bool = True) -> str:
    """Start everything and show an 'Open' button plus an embedded live view in the cell output."""
    from IPython.display import HTML, display

    url = launch(port)
    iframe = (f'<iframe src="{url}" allow="camera; autoplay; clipboard-write" '
              'style="width:100%;height:860px;border:0;border-radius:14px;margin-top:14px;'
              'background:#0b0d12"></iframe>') if embed else ""
    display(HTML(f"""
<div style="font-family:system-ui,sans-serif;padding:18px 20px;border-radius:16px;
            background:linear-gradient(135deg,#1e1b4b,#0f172a);color:#e8ebf2">
  <div style="font-size:20px;font-weight:700">👄 LipVision is live</div>
  <div style="margin:6px 0 14px;color:#a5b4fc">Open it on your laptop, click <b>Start camera</b>,
    and mouth sentences silently. Video only: the microphone is never used.</div>
  <a href="{url}" target="_blank" rel="noopener"
     style="display:inline-block;padding:12px 22px;border-radius:12px;font-weight:700;font-size:16px;
            background:linear-gradient(135deg,#818cf8,#5eead4);color:#0b0d12;text-decoration:none">
     ▶ Open LipVision (new tab)</a>
  <code style="margin-left:12px;color:#5eead4">{url}</code>
  <div style="margin-top:10px;font-size:13px;color:#94a3b8">If the embedded view below can't use your
    camera (some browsers block cameras inside notebooks), use the button above.
    The app stays live while this notebook session runs.</div>
</div>{iframe}"""))
    return url


def main(port: int = 8000):
    url = launch(port)
    bar = "=" * 64
    print(f"\n{bar}\n  LipVision is live:  {url}\n  Open it in Chrome/Edge and click 'Start camera'.\n{bar}\n", flush=True)
    try:
        while all(p.poll() is None for p in _procs):
            time.sleep(2)
        print("A background process stopped:\n" + server_log(Settings.from_env()))
    except KeyboardInterrupt:
        pass
    finally:
        stop()
