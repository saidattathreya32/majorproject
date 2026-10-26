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
    code = src + ("\nprint('SOURCE=' + get_code())\nsys.path.insert(0, APP)\n"
                  "import lipvision.launch, lipvision.server\nprint('FROM=' + lipvision.__file__)")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    assert "SOURCE=built-in copy" in out, out
    assert f"FROM={tmp_path}/majorproject/lipvision/__init__.py" in out, out


def test_cli_forwards_transcribe_args(monkeypatch):
    import lipvision.__main__ as cli
    import lipvision.transcribe as tr

    seen = {}
    monkeypatch.setattr(tr, "main", lambda: seen.setdefault("argv", sys.argv[1:]))
    cli.main(["transcribe", "a.mp4", "--beam-size", "3"])
    assert seen["argv"] == ["a.mp4", "--beam-size", "3"]
