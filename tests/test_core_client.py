import subprocess
import sys

import pytest
from napari_cellcanvas._core_client import CoreServiceClient


@pytest.mark.parametrize("url", ["http://remote.example:8001", "ftp://localhost", "not a URL"])
def test_service_connection_requires_shared_local_endpoint(url):
    with pytest.raises(ValueError):
        CoreServiceClient(url)


def test_core_client_import_is_independent_of_viewer():
    result = subprocess.run([sys.executable, "-c", "import sys; import napari_cellcanvas._core_client; assert 'qtpy' not in sys.modules; assert 'napari' not in sys.modules; assert 'cellcanvas_core' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
