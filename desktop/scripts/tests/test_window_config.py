from pathlib import Path


def test_desktop_window_is_resizable_at_compact_sizes() -> None:
    launcher = Path(__file__).resolve().parents[2] / "backend" / "launcher" / "main.py"
    source = launcher.read_text(encoding="utf-8")

    assert "resizable=True" in source
    assert "min_size=(720, 520)" in source
