from importlib.util import find_spec
from threading import Event, Thread
from types import SimpleNamespace
import socket
import time
import asyncio

import pytest

if find_spec("cellcanvas_service") is None or find_spec("cellcanvas_core") is None:
    pytest.skip("Service integration tests require service-testing", allow_module_level=True)

import numpy as np
import requests
import uvicorn
import zarr
from starlette.middleware.base import BaseHTTPMiddleware
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_service.app import create_app, default_manager
from napari.components import ViewerModel
from qtpy.QtCore import QEvent, QTimer
from qtpy.QtWidgets import QApplication, QDialogButtonBox

from napari_cellcanvas._core_client import CoreServiceManager
from napari_cellcanvas._service_dialog import ServiceDialog
from napari_cellcanvas.widget import CellCanvasWidget


@pytest.fixture
def cases(app, tmp_path, wait_until):
    items = []

    def make(depth=1):
        path = tmp_path / str(len(items))
        path.mkdir()
        indices = np.arange(depth * 5 * 4).reshape(depth, 5, 4)
        image = zarr.open_group(str(path / "image"), mode="w", zarr_format=3)
        image.create_array("0", data=np.stack([indices, indices + 100]), dimension_names=("c", "z", "y", "x"))
        image.attrs["ome"] = {"version": "0.5", "multiscales": [{
            "axes": [{"name": "c", "type": "channel"}] + [
                {"name": axis, "type": "space", "unit": "micrometer"} for axis in ("z", "y", "x")],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1, 3, 2, .5]}, {"type": "translation", "translation": [0, 7, 11, 13]}]}]}]}
        zarr.save_array(str(path / "features"), (indices % 2)[None])
        paths = [str(path / name) for name in ("image", "features", "painting", "prediction")]
        dataset = DataSet.from_paths(*paths, make_missing_datasets=True)
        root = CellCanvasWidget(viewer=ViewerModel())
        root.add_dataset(dataset)
        item = SimpleNamespace(root=root, panel=root.local_controls, dataset=dataset, paths=paths,
                               expected=indices % 2 * 2 + 1, releases=[])
        items.append(item)
        return item

    yield make
    for item in items:
        for event in item.releases:
            event.set()
        panel = item.panel
        if panel.busy and panel._worker is None and hasattr(panel.manager, "reconcile"):
            panel.manager.wait_timeout = 5
            panel.check_service()
        wait_until(lambda: not panel.busy)
        item.root.close()
        item.root.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


@pytest.fixture
def serve():
    servers = []

    def start(datasets, *, response_delay=0, **kwargs):
        app = create_app(datasets, **kwargs)
        if response_delay:
            async def delay_reply(request, call_next):
                response = await call_next(request)
                if request.method == "POST":
                    await asyncio.sleep(response_delay)
                return response
            app.add_middleware(BaseHTTPMiddleware, dispatch=delay_reply)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        servers.append((server, thread, sock))
        deadline = time.monotonic() + 5
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                pytest.fail("Core service did not start")
            time.sleep(.01)
        return f"http://127.0.0.1:{sock.getsockname()[1]}", app

    yield start
    for server, thread, sock in servers:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


def connect(case, url, wait_until, name="demo", **options):
    assert case.panel.connect_service(url, name, **options)
    wait_until(lambda: not case.panel.busy)
    assert isinstance(case.panel.manager, CoreServiceManager), case.panel.status_label.text()


def paint(case):
    layer = case.root.dataset_layers.painting
    layer.brush_size = 1
    layer.paint((0, 2, 0), 1)
    layer.paint((0, 2, 1), 3)


@pytest.mark.parametrize("depth", [1, 2])
def test_buttons_train_predict_refresh_and_reopen(cases, serve, wait_until, depth):
    case = cases(depth)
    # Open another set of core handles, as a separate service process would.
    remote = DataSet.from_paths(*case.paths)
    url, app = serve({"demo": remote})
    connect(case, url, wait_until)
    paint(case)
    original = case.dataset.labels[:].copy()
    case.panel.train_button.click()
    assert not case.root.dataset_layers.painting.editable
    wait_until(lambda: not case.panel.busy)
    assert case.panel.trained
    case.panel.predict_button.click()
    wait_until(lambda: not case.panel.busy)
    np.testing.assert_array_equal(DataSet.from_paths(*case.paths).segmentation[:], case.expected)
    np.testing.assert_array_equal(case.root.dataset_layers.prediction._data_view, case.expected[0])
    np.testing.assert_array_equal(case.dataset.labels[:], original)
    assert tuple(case.root.dataset_layers.prediction.scale) == (3, 2, .5)
    assert tuple(case.root.dataset_layers.prediction.translate) == (7, 11, 13)
    assert all(str(unit) == "micrometer" for unit in case.root.dataset_layers.prediction.units)
    assert case.panel.status_label.text() == "Predictions saved."
    assert case.root.dataset_layers.painting.editable


def test_equal_shapes_wrong_stores_rejected_and_dialog_requires_selection(cases, serve, wait_until, app):
    case, other = cases(), cases()
    url, service = serve({"demo": case.dataset, "other": other.dataset})
    manager = case.panel.manager
    case.panel.connect_service(url, "other")
    wait_until(lambda: not case.panel.busy)
    assert case.panel.manager is manager
    assert "does not match" in case.panel.status_label.text()
    assert case.root.dataset_layers.painting.editable
    dialog = ServiceDialog()
    try:
        dialog.url.setText(url)
        dialog.load_button.click()
        wait_until(lambda: dialog._worker is None)
        assert not dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
        dialog.datasets.setCurrentIndex(dialog.datasets.findData("demo"))
        assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
        dialog.url.setText(url + "/changed")
        assert not dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
    finally:
        dialog.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_timeout_keeps_painting_locked_until_receipt_completes(cases, serve, wait_until):
    case = cases()
    paint(case)
    started, release = Event(), Event()
    case.releases.append(release)
    calls = []

    class Manager:
        def fit(self):
            calls.append(True)
            started.set()
            release.wait(10)

    url, app = serve({"demo": case.dataset}, manager_factory=lambda dataset: Manager())
    connect(case, url, wait_until, wait_timeout=.05, poll_interval=.01)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(5)
    try:
        case.panel.train_button.click()
        wait_until(lambda: started.is_set() and case.panel._worker is None)
        assert ticks
        assert case.panel.busy
        assert not case.root.dataset_layers.painting.editable
        assert case.panel.check_service_button.isEnabled()
        assert not case.root.open_dataset_button.isEnabled()
        assert not case.panel.connect_service(url, "demo")
        case.panel.check_service_button.click()
        wait_until(lambda: case.panel._worker is None)
        assert case.panel.busy and not case.root.dataset_layers.painting.editable
        assert len(calls) == 1
    finally:
        release.set()
        timer.stop()
    case.panel.manager.wait_timeout = 5
    case.panel.check_service_button.click()
    wait_until(lambda: not case.panel.busy)
    assert case.panel.trained
    assert case.root.dataset_layers.painting.editable
    assert len(calls) == 1


def test_lost_post_reply_reconciles_without_repeating_training(cases, serve, wait_until):
    case = cases()
    paint(case)
    calls = []

    class Manager:
        def fit(self):
            calls.append(True)

    url, app = serve({"demo": case.dataset}, manager_factory=lambda dataset: Manager(), response_delay=1)
    connect(case, url, wait_until, timeout=(1, .2))
    case.panel.train_button.click()
    wait_until(lambda: case.panel._worker is None)
    assert case.panel.busy and not case.root.dataset_layers.painting.editable
    case.panel.check_service_button.click()
    wait_until(lambda: not case.panel.busy)
    assert case.panel.trained and len(calls) == 1
    assert case.root.dataset_layers.painting.editable


def test_known_failure_restores_editing_and_allows_retry(cases, serve, wait_until):
    case = cases()
    url, app = serve({"demo": case.dataset})
    connect(case, url, wait_until)
    case.panel.train_button.click()
    wait_until(lambda: not case.panel.busy)
    assert "failed" in case.panel.status_label.text()
    assert not case.panel.trained
    assert case.root.dataset_layers.painting.editable
    paint(case)
    case.panel.train_button.click()
    wait_until(lambda: not case.panel.busy)
    assert case.panel.trained


def test_partial_remote_prediction_refreshes_and_retries(cases, serve, wait_until):
    case = cases()
    fault = Event()
    fault.set()
    delegate = default_manager(case.dataset)

    class Manager:
        def fit(self):
            delegate.fit()

        def predict_to_segmentation(self, dataset, batch_size=65536):
            if fault.is_set():
                dataset.segmentation[0, 0, 0] = 9
                raise OSError("partial prediction")
            delegate.predict_to_segmentation(dataset, batch_size=batch_size)

    url, app = serve({"demo": case.dataset}, manager_factory=lambda dataset: Manager())
    connect(case, url, wait_until)
    paint(case)
    case.panel.train_button.click()
    wait_until(lambda: not case.panel.busy)
    case.panel.predict_button.click()
    wait_until(lambda: not case.panel.busy)
    assert "failed" in case.panel.status_label.text()
    assert case.root.dataset_layers.prediction._data_view[0, 0] == 9
    assert case.root.dataset_layers.painting.editable
    fault.clear()
    case.panel.predict_button.click()
    wait_until(lambda: not case.panel.busy)
    np.testing.assert_array_equal(case.dataset.segmentation[:], case.expected)
