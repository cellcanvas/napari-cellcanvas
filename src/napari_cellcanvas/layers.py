"""Optional napari layer adapter for a local CellCanvas dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask.array as da

if TYPE_CHECKING:
    from cellcanvas_core.data.data_set import DataSet
    from napari.components import ViewerModel
    from napari.layers import Image, Labels


@dataclass
class DatasetLayers:
    images: tuple[Image, ...]
    painting: Labels
    prediction: Labels

    def refresh_predictions(self):
        """Reload the displayed prediction slice after writes to storage."""
        self.prediction.refresh()


def add_dataset(viewer: ViewerModel, dataset: DataSet, *, name="CellCanvas") -> DatasetLayers:
    """Add lazy image channels and directly stored annotation layers.

    Painting edits the dataset labels in place. Prediction storage is shown
    as a read-only layer; call refresh_predictions after updating it.
    """
    geometry = dataset.spatial_transform
    coordinates = {
        "axis_labels": ("z", "y", "x"),
        "scale": geometry.scale,
        "translate": geometry.translation,
        "units": tuple(unit or "pixel" for unit in geometry.units),
    }
    image_data = da.asarray(dataset.image)
    images = tuple(
        viewer.add_image(image_data[channel], name=f"{name}: channel {channel}",
                         rgb=False, blending="additive", **coordinates)
        for channel in range(dataset.image.shape[0])
    )
    painting = viewer.add_labels(dataset.labels, name=f"{name}: painting", **coordinates)
    prediction = viewer.add_labels(dataset.segmentation, name=f"{name}: prediction", **coordinates)
    prediction.editable = False
    viewer.layers.selection.active = painting
    return DatasetLayers(images, painting, prediction)
