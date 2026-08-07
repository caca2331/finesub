# Re-exported: the marker decides where personal data lives, which is a
# directory-layout fact, so it is defined with the layout in finesub_bootstrap
# and shared with the CLI rather than owned by the desktop app.
from finesub_bootstrap.paths import INSTALLED_MARKER_NAME

PRODUCT_NAME = "FineSub Desktop"
MAIN_EXECUTABLE_NAME = f"{PRODUCT_NAME}.exe"
UPDATER_EXECUTABLE_NAME = f"{PRODUCT_NAME} Updater.exe"

__all__ = [
    "INSTALLED_MARKER_NAME",
    "MAIN_EXECUTABLE_NAME",
    "PRODUCT_NAME",
    "UPDATER_EXECUTABLE_NAME",
]
