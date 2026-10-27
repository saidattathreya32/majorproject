# 👄 LipVision: real-time silent lip reading on a free Kaggle GPU
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
BUNDLE = [
    "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",
]

main()
