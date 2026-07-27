from desktop.backend.common.product import (
    MAIN_EXECUTABLE_NAME,
    PRODUCT_NAME,
    UPDATER_EXECUTABLE_NAME,
)


def test_windows_product_files_use_the_approved_desktop_name() -> None:
    assert PRODUCT_NAME == "FineSub Desktop"
    assert MAIN_EXECUTABLE_NAME == "FineSub Desktop.exe"
    assert UPDATER_EXECUTABLE_NAME == "FineSub Desktop Updater.exe"
