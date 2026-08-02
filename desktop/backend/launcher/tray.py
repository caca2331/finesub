from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread
from typing import Any

from PIL import Image
import pystray


class TrayController:
    """Own the Windows notification-area icon and window visibility."""

    def __init__(self, window: Any, icon_path: Path) -> None:
        self.window = window
        self.icon_path = icon_path
        self._icon: pystray.Icon | None = None
        self._thread: Thread | None = None
        self._lock = Lock()

    def start(self) -> None:
        with self._lock:
            if self._icon is not None:
                return
            image = Image.open(self.icon_path).convert("RGBA")
            self._icon = pystray.Icon(
                "finesub-desktop",
                image,
                "FineSub Desktop",
                menu=pystray.Menu(
                    pystray.MenuItem(
                        "\u663e\u793a\u4e3b\u754c\u9762",
                        self._show_from_menu,
                        default=True,
                    ),
                    pystray.MenuItem("\u9000\u51fa", self._exit_from_menu),
                ),
            )
            self._thread = Thread(
                target=self._icon.run,
                name="finesub-system-tray",
                daemon=True,
            )
            self._thread.start()

    def hide_window(self) -> None:
        self.window.hide()

    def show_window(self) -> None:
        self.window.show()
        self.window.restore()

    def stop(self) -> None:
        with self._lock:
            icon = self._icon
            self._icon = None
            self._thread = None
        if icon is not None:
            icon.stop()

    def exit_application(self) -> None:
        self.stop()
        self.window.destroy()

    def _show_from_menu(
        self,
        _icon: pystray.Icon,
        _item: pystray.MenuItem,
    ) -> None:
        self.show_window()

    def _exit_from_menu(
        self,
        _icon: pystray.Icon,
        _item: pystray.MenuItem,
    ) -> None:
        self.exit_application()
