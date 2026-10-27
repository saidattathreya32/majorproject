import ast
import base64
import importlib.util
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _builder():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "scripts" / "build_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell_source():
    nb = json.loads((ROOT / "kaggle" / "LipVision_Kaggle.ipynb").read_text())
    assert len(nb["cells"]) == 1
    cell = nb["cells"][0]
    # markdown executed as code caused "SyntaxError: invalid decimal literal" on Kaggle
    assert cell["cell_type"] == "code"
    return "".join(cell["source"])


def test_notebook_is_single_valid_code_cell():
    src = _cell_source()
    ast.parse(src)
    # os.chdir alone left `lipvision` unimportable in the kernel (ModuleNotFoundError)
    assert "sys.path.insert(0, APP)" in src
    # a private repo made git prompt for a username and fail
    assert 'GIT_TERMINAL_PROMPT="0"' in src


def test_notebook_and_cell_file_are_up_to_date():
    """Fails if lipvision/ changed without re-running scripts/build_notebook.py."""
    expected = _builder().build_cell()
    assert _cell_source() == expected
    assert (ROOT / "kaggle" / "kaggle_cell.py").read_text() == expected


def test_embedded_bundle_contains_the_app():
    ns = {}
    src = _cell_source()
    exec(src[src.index("BUNDLE = ["):src.rindex("main()")], ns)
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(ns["BUNDLE"]))))
    names = set(z.namelist())
    for f in ("lipvision/launch.py", "lipvision/server.py", "lipvision/web/app.js", "requirements-kaggle.txt"):
        assert f in names
    assert z.read("lipvision/server.py") == (ROOT / "lipvision" / "server.py").read_bytes()


def test_private_repo_falls_back_to_embedded_copy(tmp_path):
    """Run the real cell's get_code() against an unreachable repo."""
    src = _cell_source()
    src = src.replace('REPO = "https://github.com/saidattathreya32/majorproject"',
                      f'REPO = "file://{tmp_path}/does-not-exist"')
    src = src[:src.rindex("main()")]  # define everything, don't launch
    env = dict(os.environ)
    code = src + ("\nfor c in candidates():\n    try:\n        print('SOURCE=' + get_code(*c)); break\n    except Exception:\n        pass\nsys.path.insert(0, APP)\n"
                  "import lipvision.launch, lipvision.server\nprint('FROM=' + lipvision.__file__)")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    assert "SOURCE=built-in copy" in out, out
    assert f"FROM={tmp_path}/majorproject/lipvision/__init__.py" in out, out


def _fake_release_repo(path, good_tags, bad_tags):
    """A git repo whose tags each hold a tiny lipvision package; bad tags fail to launch."""
    def sh(*a):
        subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True)

    path.mkdir()
    sh("init", "-q", "-b", "main")
    (path / "requirements-kaggle.txt").write_text("")
    pkg = path / "lipvision"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for tag in sorted(good_tags + bad_tags, key=lambda t: [int(x) for x in t[1:].split(".")]):
        body = (f"raise RuntimeError('broken release {tag}')" if tag in bad_tags
                else f"print('LAUNCHED {tag}'); return 'url-{tag}'")
        (pkg / "launch.py").write_text(f"def launch_in_notebook():\n    {body}\n")
        sh("add", "-A")
        sh("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", tag)
        sh("tag", tag)
    return path


def _run_cell_main(tmp_path, repo_url, extra=""):
    src = _cell_source()
    src = src.replace('REPO = "https://github.com/saidattathreya32/majorproject"', f'REPO = "{repo_url}"')
    src = src.replace('WORK = "/kaggle/working"', f'WORK = "{tmp_path}/work"')
    src = src[:src.rindex("main()")] + extra + "\nmain()\n"
    (tmp_path / "work").mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, "-c", src], cwd=tmp_path, capture_output=True, text=True,
                       env=dict(os.environ))
    return r.returncode, r.stdout + r.stderr


def test_broken_newest_release_rolls_back_to_previous(tmp_path):
    repo = _fake_release_repo(tmp_path / "remote", good_tags=["v1.0"], bad_tags=["v1.1"])
    code, out = _run_cell_main(tmp_path, f"file://{repo}")
    assert code == 0, out
    assert "v1.1 failed" in out and "broken release v1.1" in out, out
    assert "LAUNCHED v1.0" in out, out
    assert "Running v1.0" in out, out


def test_newest_good_release_is_used_without_fallback(tmp_path):
    repo = _fake_release_repo(tmp_path / "remote", good_tags=["v1.0", "v1.2"], bad_tags=[])
    code, out = _run_cell_main(tmp_path, f"file://{repo}")
    assert code == 0, out
    assert "LAUNCHED v1.2" in out and "LAUNCHED v1.0" not in out, out


def test_pinned_version_is_honoured(tmp_path):
    repo = _fake_release_repo(tmp_path / "remote", good_tags=["v1.0", "v1.1"], bad_tags=[])
    code, out = _run_cell_main(tmp_path, f"file://{repo}", extra='\nPIN = "v1.0"\n')
    assert code == 0, out
    assert "LAUNCHED v1.0" in out and "LAUNCHED v1.1" not in out, out


def test_cli_forwards_transcribe_args(monkeypatch):
    import lipvision.__main__ as cli
    import lipvision.transcribe as tr

    seen = {}
    monkeypatch.setattr(tr, "main", lambda: seen.setdefault("argv", sys.argv[1:]))
    cli.main(["transcribe", "a.mp4", "--beam-size", "3"])
    assert seen["argv"] == ["a.mp4", "--beam-size", "3"]
