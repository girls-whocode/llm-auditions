import subprocess
from pathlib import Path


def test_script_lists_only_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "dir1").mkdir()
    script = Path("solution.sh")
    script.chmod(0o755)
    res = subprocess.run(["bash", str(script), str(tmp_path)], capture_output=True, text=True)
    assert res.returncode == 0
    out = res.stdout
    assert "a.txt" in out or "b.txt" in out
    assert "dir1" not in out
