PRODUCT_NAME = "FineSub Desktop"
MAIN_EXECUTABLE_NAME = f"{PRODUCT_NAME}.exe"
UPDATER_EXECUTABLE_NAME = f"{PRODUCT_NAME} Updater.exe"

# Written next to the executable by the Inno Setup installer (and only by it):
# its presence is what separates an installed copy (personal data in
# %LOCALAPPDATA%\FineSub) from a portable one (everything beside the exe).
# The updater preserves it across full updates; update payloads never ship it.
INSTALLED_MARKER_NAME = "installed.marker"
