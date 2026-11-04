from importlib.util import find_spec
from threading import Event, get_ident
from unittest.mock import Mock

import pytest

if find_spec("cellcanvas_core") is None:
    pytest.skip("Dataset loading requires the local extra", allow_module_level=True)

import numpy as np
import zarr
from napari.components import ViewerModel
from qtpy.QtCore import QEvent, QTimer
from qtpy.QtWidgets import QApplication, QDialogButtonBox

import napari_cellcanvas._dataset as dataset_module
import napari_cellcanvas.widget as widget_module
from napari_cellcanvas._dataset import LocalDatasetDialog
from napari_cellcanvas.widget import CellCanvasWidget


@pytest.fixture
def host(app, wait_until):
    widget = CellCanvasWidget(viewer=ViewerModel())
    yield widget
    wait_until(lambda: not widget._dataset_loader.busy and not widget.local_controls.busy)
    widget.close()
    # Destroy test-owned Qt objects on the GUI thread before later Dask work
    # can trigger Python garbage collection on a worker thread.
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.fixture
def stores(tmp_path):
    def make(depth=1, feature_axis=0):
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        root.mkdir()
        indices = np.arange(depth * 5 * 4).reshape(depth, 5, 4)
        paths = {key: str(root / name) for key, name in (
            ("image_path", "image"), ("features_path", "features"),
            ("labels_path", "painting"), ("segmentation_path", "prediction"))}
        zarr.save_array(paths["image_path"], np.stack([indices, indices + 100]))
        features = (indices % 2)[None]
        if feature_axis == -1:
            features = np.moveaxis(features, 0, -1)
        zarr.save_array(paths["features_path"], features)
        return dict(paths, feature_axis=feature_axis, make_missing_datasets=True), indices
    return make


@pytest.mark.parametrize("depth,feature_axis", [(1, 0), (2, 0), (1, -1)])
def test_dialog_load_paint_train_predict_reopen(host, stores, wait_until, monkeypatch, depth, feature_axis):
    options, indices = stores(depth, feature_axis)
    dialog = LocalDatasetDialog(host)
    for key, field in dialog.paths.items():
        field.setText(options[key])
    dialog.feature_order.setCurrentIndex(1 if feature_axis == -1 else 0)
    dialog.create_annotations.setChecked(True)
    assert dialog.buttons.button(QDialogButtonBox.Open).isEnabled()
    # Exercise the widget button with a real dialog and selected stores.
    monkeypatch.setattr(widget_module, "LocalDatasetDialog", lambda parent: dialog)
    monkeypatch.setattr(dialog, "exec", lambda: 1)
    host.open_dataset_button.click()
    assert host._dataset_loader.busy
    assert not host.local_controls.isEnabled()
    wait_until(lambda: not host._dataset_loader.busy)
    assert host.status_label.text() == "Loaded image"
    assert len(host.viewer.layers) == 4
    panel, layers = host.local_controls, host.dataset_layers
    assert panel.dataset.feature_axis == feature_axis
    layers.painting.brush_size = 1
    layers.painting.paint((0, 2, 0), 1)
    layers.painting.paint((0, 2, 1), 3)
    panel.train_button.click()
    assert not host.open_dataset_button.isEnabled()
    wait_until(lambda: not panel.busy)
    panel.predict_button.click()
    wait_until(lambda: not panel.busy)
    np.testing.assert_array_equal(zarr.open_array(options["segmentation_path"], mode="r")[:],
                                  indices % 2 * 2 + 1)
    assert host.open_dataset_button.isEnabled()


def test_failure_preserves_dataset_and_replacement_preserves_unrelated_layers(host, stores, wait_until):
    first, _ = stores()
    host.open_dataset(**first)
    wait_until(lambda: not host._dataset_loader.busy)
    original = host.dataset_layers
    manager = host.local_controls.manager
    extra = host.viewer.add_points([[0, 0, 0]], name="unrelated")
    invalid, _ = stores()
    invalid["image_path"] += "-missing"
    host.open_dataset(**invalid)
    wait_until(lambda: not host._dataset_loader.busy)
    assert "Could not open local dataset" in host.status_label.text()
    assert host.dataset_layers is original
    assert host.local_controls.manager is manager
    assert len(host.viewer.layers) == 5
    second, _ = stores(2)
    host.open_dataset(**second)
    wait_until(lambda: not host._dataset_loader.busy)
    assert host.dataset_layers is not original
    assert extra in host.viewer.layers
    assert all(layer not in host.viewer.layers for layer in (*original.images, original.painting, original.prediction))
    assert len(host.viewer.layers) == 5


def test_display_failure_rolls_back_new_layers(host, stores, wait_until, monkeypatch):
    options, _ = stores()
    host.open_dataset(**options)
    wait_until(lambda: not host._dataset_loader.busy)
    original = tuple(host.viewer.layers)
    active = host.viewer.layers.selection.active
    manager = host.local_controls.manager
    monkeypatch.setattr(ViewerModel, "add_labels", Mock(side_effect=RuntimeError("display fault")))
    host.open_dataset(**options)
    wait_until(lambda: not host._dataset_loader.busy)
    assert tuple(host.viewer.layers) == original
    assert host.viewer.layers.selection.active is active
    assert host.local_controls.manager is manager
    assert "display fault" in host.status_label.text()
    assert host.local_controls.isEnabled()


def test_slow_load_is_responsive_and_blocks_replacement(host, stores, wait_until, monkeypatch):
    options, _ = stores()
    started, release = Event(), Event()
    threads = []
    real_load = dataset_module.load_dataset

    def delayed(options):
        threads.append(get_ident())
        started.set()
        release.wait(3)
        return real_load(options)

    monkeypatch.setattr(dataset_module, "load_dataset", delayed)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(5)
    try:
        assert host.open_dataset(**options)
        wait_until(lambda: started.is_set() and len(ticks) >= 5)
        assert not host.open_dataset_button.isEnabled()
        assert not host.open_dataset(**options)
        with pytest.raises(RuntimeError, match="loading"):
            host.add_dataset(None)
        assert threads == [threads[0]] and threads[0] != get_ident()
    finally:
        release.set()
        timer.stop()
    wait_until(lambda: not host._dataset_loader.busy)
    assert host.dataset_layers is not None


def test_close_discards_pending_load(host, stores, wait_until, monkeypatch):
    options, _ = stores()
    started, release = Event(), Event()
    real_load = dataset_module.load_dataset

    def delayed(options):
        started.set()
        release.wait(3)
        return real_load(options)

    monkeypatch.setattr(dataset_module, "load_dataset", delayed)
    try:
        host.open_dataset(**options)
        wait_until(started.is_set)
        host.close()
    finally:
        release.set()
    wait_until(lambda: not host._dataset_loader.busy)
    assert host.dataset_layers is None
    assert len(host.viewer.layers) == 0


def test_create_is_opt_in_and_existing_painting_survives(host, stores, wait_until):
    options, indices = stores()
    options["make_missing_datasets"] = False
    host.open_dataset(**options)
    wait_until(lambda: not host._dataset_loader.busy)
    assert host.dataset_layers is None
    options["make_missing_datasets"] = True
    painting = zarr.open_array(options["labels_path"], mode="w", shape=indices.shape, dtype="i4")
    painting[0, 2, 0] = 7
    host.open_dataset(**options)
    wait_until(lambda: not host._dataset_loader.busy)
    assert host.dataset_layers.painting.data[0, 2, 0] == 7
    assert host.local_controls.isEnabled()
