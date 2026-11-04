from importlib.util import find_spec
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

if find_spec("cellcanvas_core") is None:
    pytest.skip("Local controls require the local extra", allow_module_level=True)

import numpy as np
import zarr
from napari.components import ViewerModel
from qtpy.QtCore import QEvent, QTimer
from qtpy.QtWidgets import QApplication
from sklearn.tree import DecisionTreeClassifier

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager
from napari_cellcanvas._local import LocalSegmentationControls, _LayerGuard
from napari_cellcanvas.layers import add_dataset
from napari_cellcanvas.widget import CellCanvasWidget


@pytest.fixture
def case_factory(app, tmp_path, wait_until):
    cases = []

    def make(depth=1, manager_factory=None, host=False, default=False, trained=False):
        path = tmp_path / str(len(cases))
        path.mkdir()
        indices = np.arange(depth * 5 * 4).reshape(depth, 5, 4)
        labels = zarr.open_array(str(path / "labels"), mode="w", shape=indices.shape, dtype="i4")
        output = zarr.open_array(str(path / "output"), mode="w", shape=indices.shape, dtype="i4")
        dataset = DataSet(np.stack([indices, indices + 100]), {"parity": (indices % 2)[None]}, labels, output)
        manager = manager_factory(dataset) if manager_factory else SemanticSegmentationManager(
            DataManager([dataset]), DecisionTreeClassifier(random_state=0))
        viewer = ViewerModel()
        if host:
            root = CellCanvasWidget(viewer=viewer)
            layers = root.add_dataset(dataset, manager=None if default else manager, trained=trained)
            panel = root.local_controls
            manager = panel.manager
        else:
            layers = add_dataset(viewer, dataset)
            panel = root = LocalSegmentationControls()
            panel.set_dataset(dataset, layers, manager, trained=trained)
        destroyed = []
        root.destroyed.connect(lambda: destroyed.append(True))
        result = SimpleNamespace(dataset=dataset, manager=manager, layers=layers, viewer=viewer,
                                 panel=panel, root=root, path=path, indices=indices,
                                 destroyed=destroyed, releases=[])
        cases.append(result)
        return result

    yield make
    for case in cases:
        for event in case.releases:
            event.set()
        if case.destroyed:
            wait_until(lambda: case.layers.painting.editable)
        else:
            wait_until(lambda: not case.panel.busy)
            case.root.close()


def paint_two_classes(case):
    case.layers.painting.brush_size = 1
    case.layers.painting.paint((0, 2, 0), 1)
    case.layers.painting.paint((0, 2, 1), 3)


@pytest.mark.parametrize("depth", [1, 2])
def test_buttons_train_predict_save_and_reopen(case_factory, wait_until, depth):
    case = case_factory(depth, host=True, default=True)
    paint_two_classes(case)
    case.layers.painting.mode = "paint"
    original_painting = case.dataset.labels[:].copy()
    assert case.panel.train_button.isEnabled()
    assert not case.panel.predict_button.isEnabled()
    case.panel.train_button.click()
    assert case.panel.busy
    assert not case.layers.painting.editable
    wait_until(lambda: not case.panel.busy)
    assert case.panel.trained
    assert case.panel.predict_button.isEnabled()
    assert case.layers.painting.mode == "paint"
    case.panel.predict_button.click()
    assert case.panel.busy
    wait_until(lambda: not case.panel.busy)
    expected = case.indices % 2 * 2 + 1
    np.testing.assert_array_equal(zarr.open_array(str(case.path / "output"), mode="r")[:], expected)
    np.testing.assert_array_equal(case.layers.prediction._data_view, expected[0])
    np.testing.assert_array_equal(case.dataset.labels[:], original_painting)
    assert case.panel.status_label.text() == "Predictions saved."
    assert case.layers.painting.editable


def test_empty_painting_failure_recovers(case_factory, wait_until):
    case = case_factory()
    case.panel.train_button.click()
    wait_until(lambda: not case.panel.busy)
    assert "failed" in case.panel.status_label.text()
    assert not case.panel.predict_button.isEnabled()
    assert case.panel.train_button.isEnabled()
    assert case.layers.painting.editable
    paint_two_classes(case)
    case.panel.train_button.click()
    wait_until(lambda: not case.panel.busy)
    assert case.panel.trained


def test_slow_training_keeps_gui_responsive_and_blocks_dataset_switch(case_factory, wait_until):
    started, release = Event(), Event()
    threads = []

    class Manager:
        def fit(self):
            threads.append(get_ident())
            started.set()
            release.wait(2)

    case = case_factory(manager_factory=lambda dataset: Manager(), host=True)
    case.releases.append(release)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(5)
    case.panel.train_button.click()
    wait_until(lambda: started.is_set() and len(ticks) >= 5)
    assert not case.panel.train_button.isEnabled()
    assert not case.panel.predict_button.isEnabled()
    before = len(case.viewer.layers)
    with pytest.raises(RuntimeError, match="finish"):
        case.root.add_dataset(case.dataset)
    assert len(case.viewer.layers) == before
    case.panel.train()
    assert len(threads) == 1 and threads[0] != get_ident()
    release.set()
    wait_until(lambda: not case.panel.busy)
    timer.stop()


def test_injected_pretrained_manager_is_used_without_retraining(case_factory, wait_until):
    case = case_factory()
    paint_two_classes(case)
    case.manager.fit()
    second = LocalSegmentationControls()
    second.set_dataset(case.dataset, case.layers, case.manager, trained=True)
    assert second.manager is case.manager
    second.predict_button.click()
    wait_until(lambda: not second.busy)
    np.testing.assert_array_equal(case.dataset.segmentation[:], case.indices % 2 * 2 + 1)
    second.close()


def test_failed_prediction_refreshes_partial_writes_and_recovers(case_factory, wait_until):
    class Manager:
        def predict_to_segmentation(self, dataset):
            dataset.segmentation[0, 0, 0] = 9
            raise OSError("storage failure")

    case = case_factory(manager_factory=lambda dataset: Manager(), trained=True)
    assert not case.panel.train_button.isEnabled()
    case.panel.predict_button.click()
    wait_until(lambda: not case.panel.busy)
    assert "Prediction failed" in case.panel.status_label.text()
    assert case.layers.prediction._data_view[0, 0] == 9
    assert case.panel.predict_button.isEnabled()
    assert case.layers.painting.editable
    np.testing.assert_array_equal(case.dataset.labels[:], 0)


def test_deleted_controls_restore_editing_when_work_finishes(case_factory, wait_until, app):
    started, release = Event(), Event()

    class Manager:
        def fit(self):
            started.set()
            release.wait(2)

    case = case_factory(manager_factory=lambda dataset: Manager())
    case.releases.append(release)
    case.layers.painting.mode = "paint"
    case.panel.train_button.click()
    wait_until(started.is_set)
    case.root.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    wait_until(lambda: bool(case.destroyed))
    assert not case.layers.painting.editable
    release.set()
    wait_until(lambda: case.layers.painting.editable)
    assert case.layers.painting.mode == "paint"
    assert not _LayerGuard._active
