"""LipVision command line.

    python -m lipvision                 live lip reading on this computer (opens your browser)
    python -m lipvision window          live lip reading in an OpenCV window (no browser)
    python -m lipvision cloud           Kaggle/Colab/VM: server + public https link
    python -m lipvision transcribe F..  transcribe video files
    python -m lipvision download        pre-download model code and weights
"""

import argparse
import logging
import sys
import threading
import time
import urllib.request
import webbrowser


def _open_when_ready(url: str):
    def run():
        for _ in range(1800):
            try:
                with urllib.request.urlopen(url + "api/health", timeout=2) as r:
                    if b'"ready":true' in r.read().replace(b" ", b""):
                        print(f"\n  LipVision is ready: {url}  (opening your browser)\n", flush=True)
                        webbrowser.open(url)
                        return
            except OSError:
                pass
            time.sleep(1)
    threading.Thread(target=run, daemon=True).start()


def local(port: int, open_browser: bool):
    import uvicorn

    from .config import Settings
    from .server import create_app

    s = Settings.from_env(host="127.0.0.1", port=port)
    url = f"http://localhost:{port}/"
    print(f"Loading the model (first run downloads ~1 GB of weights). The app will open at {url}", flush=True)
    if open_browser:
        _open_when_ready(url)
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="warning", ws_max_size=4 << 20)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m lipvision", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    lp = sub.add_parser("local", help="browser app on this computer (default)")
    lp.add_argument("--port", type=int, default=8000)
    lp.add_argument("--no-browser", action="store_true")
    sub.add_parser("window", help="OpenCV window app", add_help=False)
    cp = sub.add_parser("cloud", help="server + public https tunnel")
    cp.add_argument("--port", type=int, default=8000)
    sub.add_parser("transcribe", help="transcribe video files", add_help=False)
    sub.add_parser("download", help="download model code and weights")
    argv = sys.argv[1:] if argv is None else argv
    a, rest = p.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.cmd in (None, "local"):
        local(getattr(a, "port", 8000), not getattr(a, "no_browser", False))
    elif a.cmd == "window":
        from . import local_live
        sys.argv = ["lipvision window", *rest]
        local_live.main()
    elif a.cmd == "transcribe":
        from . import transcribe
        sys.argv = ["lipvision transcribe", *rest]
        transcribe.main()
    elif a.cmd == "cloud":
        from . import launch
        launch.main(a.port)
    elif a.cmd == "download":
        from . import assets
        assets.main()


if __name__ == "__main__":
    main()
