from importlib.util import find_spec

import pytest

if find_spec("cellcanvas_core") is None:
    pytest.skip("Local dataset tests require the local extra", allow_module_level=True)

import dask.array as da
import numpy as np
import zarr
from sklearn.tree import DecisionTreeClassifier

from napari.components import ViewerModel

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from napari_cellcanvas.layers import add_dataset
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager


def make_dataset(tmp_path, depth=1, zarr_format=3):
    indices = np.arange(depth * 5 * 4).reshape(depth, 5, 4)
    image = zarr.open_group(str(tmp_path / "image"), mode="w", zarr_format=zarr_format)
    options = {"dimension_names": ("c", "z", "y", "x")} if zarr_format == 3 else {}
    image.create_array("image", data=np.stack([indices, indices + 100]), **options)
    multiscale = {
        "axes": [{"name": "c", "type": "channel"}] + [
            {"name": name, "type": "space", "unit": "micrometer"} for name in ("z", "y", "x")
        ],
        "datasets": [{"path": "image", "coordinateTransformations": [
            {"type": "scale", "scale": [1, 3, 2, 0.5]},
            {"type": "translation", "translation": [0, 7, 11, 13]},
        ]}],
    }
    if zarr_format == 3:
        image.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale]}
    else:
        multiscale["version"] = "0.4"
        image.attrs["multiscales"] = [multiscale]
    paths = [str(tmp_path / name) for name in ("image", "features", "labels", "prediction")]
    zarr.save_array(paths[1], (indices % 2)[None], zarr_format=zarr_format)
    for path in paths[2:]:
        zarr.save_array(path, np.zeros_like(indices, dtype="i4"), zarr_format=zarr_format)
    return DataSet.from_paths(*paths), indices


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_paint_train_predict_refresh_and_reopen(tmp_path, depth, zarr_format):
    dataset, indices = make_dataset(tmp_path, depth, zarr_format)
    original = dataset.image[:].copy()
    viewer = ViewerModel()
    layers = add_dataset(viewer, dataset)
    assert len(layers.images) == 2
    assert len(viewer.layers) == 4
    assert layers.painting.data is dataset.labels
    assert layers.prediction.data is dataset.segmentation
    assert layers.painting.editable
    assert not layers.prediction.editable
    assert viewer.layers.selection.active is layers.painting
    for channel, layer in enumerate(layers.images):
        assert isinstance(layer.data, da.Array)
        assert layer.data.shape == indices.shape
        assert not layer.rgb  # X=4 must not be mistaken for an RGBA axis.
        np.testing.assert_array_equal(layer.data.compute(), original[channel])
    for layer in (*layers.images, layers.painting, layers.prediction):
        assert layer.axis_labels == ("z", "y", "x")
        np.testing.assert_array_equal(layer.data_to_world((0, 2, 2)), (7, 15, 14))
        assert all(str(unit) == "micrometer" for unit in layer.units)
    viewer.dims.set_point(0, 0)
    layers.painting.brush_size = 1
    layers.painting.paint((0, 2, 0), 1)
    layers.painting.paint((0, 2, 1), 3)
    reopened_labels = zarr.open_array(str(tmp_path / "labels"), mode="r")
    assert reopened_labels[0, 2, 0] == 1
    assert reopened_labels[0, 2, 1] == 3
    np.testing.assert_array_equal(layers.prediction._data_view, 0)
    manager = SemanticSegmentationManager(DataManager([dataset]), DecisionTreeClassifier(random_state=0))
    manager.fit()
    manager.predict_to_segmentation(dataset, batch_size=3)
    layers.refresh_predictions()
    expected = indices % 2 * 2 + 1
    np.testing.assert_array_equal(layers.prediction._data_view, expected[0])
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "prediction"), mode="r")[:], expected)
    np.testing.assert_array_equal(dataset.image[:], original)
    assert np.count_nonzero(dataset.labels[:]) == 2


def test_brush_and_undo_write_through_to_storage(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    layers = add_dataset(ViewerModel(), dataset)
    layers.painting.brush_size = 3
    layers.painting.paint((0, 2, 2), 2)
    painted = dataset.labels[:].copy()
    assert np.count_nonzero(painted) > 1
    layers.painting.undo()
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "labels"), mode="r")[:], 0)
    layers.painting.redo()
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "labels"), mode="r")[:], painted)


def test_image_without_physical_metadata_uses_pixel_coordinates():
    dataset = DataSet(np.zeros((1, 1, 3, 4)), {"features": np.zeros((1, 1, 3, 4))},
                      np.zeros((1, 3, 4), dtype="i4"), np.zeros((1, 3, 4), dtype="i4"))
    layers = add_dataset(ViewerModel(), dataset)
    for layer in (*layers.images, layers.painting, layers.prediction):
        assert all(str(unit) == "pixel" for unit in layer.units)
        np.testing.assert_array_equal(layer.data_to_world((0, 1, 2)), (0, 1, 2))


def test_widget_hosts_dataset_layers(tmp_path, app):
    from napari_cellcanvas.widget import CellCanvasWidget
    dataset, _ = make_dataset(tmp_path)
    viewer = ViewerModel()
    widget = CellCanvasWidget(viewer=viewer)
    layers = widget.add_dataset(dataset, name="Test data")
    assert widget.dataset_layers is layers
    assert len(viewer.layers) == 4
    assert layers.painting.data is dataset.labels
    assert widget.status_label.text() == "Loaded Test data"
    widget.close()
