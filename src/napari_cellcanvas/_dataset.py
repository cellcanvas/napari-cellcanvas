"""Local dataset selection and background loading for the napari frontend."""

from pathlib import Path

from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
)
from superqt.utils import create_worker

from ._local import default_manager


def load_dataset(options):
    """Delegate storage and shape validation to the headless core."""
    try:
        from cellcanvas_core.data.data_set import DataSet
    except ImportError as error:
        raise ImportError("Local datasets require napari-cellcanvas[qt,local].") from error
    dataset = DataSet.from_paths(**options)
    return dataset, default_manager(dataset)


class LocalDatasetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Local Dataset")
        layout = QFormLayout(self)
        self.paths = {}
        for key, label in (
            ("image_path", "Image (CZYX)"),
            ("features_path", "Features"),
            ("labels_path", "Painting (ZYX)"),
            ("segmentation_path", "Prediction (ZYX)"),
        ):
            row = QHBoxLayout()
            field = QLineEdit()
            self.paths[key] = field
            row.addWidget(field)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda checked=False, field=field: self._browse(field))
            row.addWidget(browse)
            layout.addRow(label, row)
        self.feature_order = QComboBox()
        self.feature_order.addItem("DZYX", 0)
        self.feature_order.addItem("ZYXD", -1)
        layout.addRow("Feature order", self.feature_order)
        self.image_level = QSpinBox()
        self.image_level.setRange(0, 999)
        layout.addRow("Image level", self.image_level)
        self.create_annotations = QCheckBox("Create missing painting and prediction stores")
        layout.addRow(self.create_annotations)
        self.message = QLabel("Choose Zarr directories. Enter new annotation paths to create stores.")
        self.message.setWordWrap(True)
        layout.addRow(self.message)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)
        for field in self.paths.values():
            field.textChanged.connect(self._validate)
        self._validate()

    def _browse(self, field):
        path = QFileDialog.getExistingDirectory(self, "Choose Zarr directory", field.text())
        if path:
            field.setText(path)

    def _validate(self):
        self.buttons.button(QDialogButtonBox.Open).setEnabled(
            all(field.text().strip() for field in self.paths.values()))

    def options(self):
        return {
            **{key: str(Path(field.text().strip()).expanduser()) for key, field in self.paths.items()},
            "feature_axis": self.feature_order.currentData(),
            "image_level": self.image_level.value(),
            "make_missing_datasets": self.create_annotations.isChecked(),
        }

    def accept(self):
        if not all(field.text().strip() for field in self.paths.values()):
            return
        options = self.options()
        if Path(options["labels_path"]).resolve() == Path(options["segmentation_path"]).resolve():
            self.message.setText("Painting and prediction must use separate stores.")
            return
        super().accept()


class DatasetLoader(QObject):
    """Deliver results on the GUI thread; discard them after the widget closes."""

    loaded = Signal(object)
    failed = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.busy = False
        self._closed = False
        self._worker = None

    def start(self, options):
        if self.busy or self._closed:
            return False
        self.busy = True
        self.busy_changed.emit(True)
        try:
            worker = create_worker(lambda options=dict(options): load_dataset(options),
                                   _start_thread=False, _ignore_errors=True)
            worker.returned.connect(self._loaded, Qt.QueuedConnection)
            worker.errored.connect(self._failed, Qt.QueuedConnection)
            worker.finished.connect(self._finished, Qt.QueuedConnection)
            self._worker = worker
            worker.start()
        except Exception as error:
            self._failed(error)
            self._finished()
        return True

    @Slot(object)
    def _loaded(self, result):
        if not self._closed:
            self.loaded.emit(result)

    @Slot(object)
    def _failed(self, error):
        if not self._closed:
            self.failed.emit(error)

    @Slot()
    def _finished(self):
        self.busy = False
        self._worker = None
        if not self._closed:
            self.busy_changed.emit(False)

    def close(self):
        # Core loading (including requested store creation) can still finish.
        self._closed = True
