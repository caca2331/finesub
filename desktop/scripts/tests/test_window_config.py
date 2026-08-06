from pathlib import Path


def test_desktop_window_is_resizable_at_compact_sizes() -> None:
    launcher = Path(__file__).resolve().parents[2] / "backend" / "launcher" / "main.py"
    source = launcher.read_text(encoding="utf-8")

    assert "resizable=True" in source
    assert "min_size=(720, 520)" in source
    assert "enable_native_window_resize(window)" in source
    assert "window.events.shown.wait(10)" in source
    assert "WM_NCHITTEST" in source
    assert "SetWindowSubclass" in source
    assert "native.Invoke" in source
    assert "17 if on_bottom and on_right" in source
    assert "HTCAPTION" in source
