import pytest
import time
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def wait_until(app):
    def wait(condition, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition():
            app.processEvents()
            if time.monotonic() > deadline:
                pytest.fail("Timed out waiting for local operation")
            time.sleep(0.001)
        app.processEvents()
    return wait
