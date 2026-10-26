import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_notebook_is_code_only_and_valid_python():
    nb = json.loads((ROOT / "kaggle" / "LipVision_Kaggle.ipynb").read_text())
    assert nb["cells"], "empty notebook"
    for cell in nb["cells"]:
        # markdown executed as code caused "SyntaxError: invalid decimal literal" on Kaggle
        assert cell["cell_type"] == "code"
        src = "".join(cell["source"])
        ast.parse(src)
        # os.chdir alone left `lipvision` unimportable in the kernel
        assert "sys.path.insert(0, APP)" in src


def test_cell_file_matches_notebook():
    nb = json.loads((ROOT / "kaggle" / "LipVision_Kaggle.ipynb").read_text())
    cell = (ROOT / "kaggle" / "kaggle_cell.py").read_text()
    assert "".join(nb["cells"][0]["source"]) == cell


def test_cli_forwards_transcribe_args(monkeypatch):
    import lipvision.__main__ as cli
    import lipvision.transcribe as tr

    seen = {}
    monkeypatch.setattr(tr, "main", lambda: seen.setdefault("argv", __import__("sys").argv[1:]))
    cli.main(["transcribe", "a.mp4", "--beam-size", "3"])
    assert seen["argv"] == ["a.mp4", "--beam-size", "3"]
