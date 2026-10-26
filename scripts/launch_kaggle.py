"""Start the LipVision server on Kaggle/Colab and expose it with a free HTTPS tunnel.

    python scripts/launch_kaggle.py [--port 8000] [--tunnel cloudflared|none]

Prints a public https:// URL. Open it on your laptop: the browser streams your
webcam (video only) to the cloud GPU and shows the transcript live.
Cloudflare quick tunnels need no account and support WebSockets.
"""

import argparse
import json
import os
import platform
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(os.environ.get("LIPVISION_HOME", ROOT / ".cache")) / "cloudflared"


def get_cloudflared() -> Path:
    if BIN.exists():
        return BIN
    arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
    BIN.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading cloudflared…")
    urllib.request.urlretrieve(url, BIN)
    BIN.chmod(BIN.stat().st_mode | stat.S_IEXEC)
    return BIN


def wait_healthy(port: int, server: subprocess.Popen, timeout: float = 900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if server.poll() is not None:
            sys.exit("Server exited during start-up; see the log above.")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as r:
                info = json.load(r)
            if info["ready"]:
                return info
            if info["error"]:
                server.terminate()
                sys.exit(f"Model failed to load:\n{info['error']}")
        except OSError:
            pass
        time.sleep(2)
    sys.exit("Server did not become healthy in time.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tunnel", default="cloudflared", choices=["cloudflared", "none"])
    a = p.parse_args()

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    server = subprocess.Popen([sys.executable, "-m", "lipvision.server", "--port", str(a.port)],
                              cwd=ROOT, env=env)
    print("Loading model (first run downloads ~1 GB of weights)…")
    info = wait_healthy(a.port, server)
    print(f"Model ready on {info['device']} (fp16={info['fp16']}, beam={info['beam']})")

    tunnel = None
    if a.tunnel == "cloudflared":
        tunnel = subprocess.Popen([str(get_cloudflared()), "tunnel", "--no-autoupdate", "--url",
                                   f"http://127.0.0.1:{a.port}"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        url = None
        for line in tunnel.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                url = m.group(0)
                break
        # keep draining cloudflared's log so it never blocks on a full pipe
        threading.Thread(target=lambda: [None for _ in tunnel.stdout], daemon=True).start()
        bar = "=" * 64
        print(f"\n{bar}\n  LipVision is live:  {url}\n  Open it in Chrome/Edge on your laptop and click 'Start camera'.\n{bar}\n",
              flush=True)
    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (tunnel, server):
            if proc and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
