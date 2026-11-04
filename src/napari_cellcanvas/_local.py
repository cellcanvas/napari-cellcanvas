"""Qt controls that delegate local segmentation to a headless core manager."""

from qtpy.QtCore import QObject, Qt, Slot
from qtpy.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from superqt.utils import create_worker


def default_manager(dataset):
    from cellcanvas_core.data.data_manager import DataManager
    from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager
    from sklearn.ensemble import RandomForestClassifier

    return SemanticSegmentationManager(
        DataManager([dataset]), RandomForestClassifier(random_state=0, n_jobs=1)
    )


class _LayerGuard(QObject):
    """Restore painting even if the controls are deleted during core work."""

    _active = set()

    def __init__(self, layers, predict):
        super().__init__()
        self.layers = layers
        self.predict = predict
        self.editable = layers.painting.editable
        self.mode = layers.painting.mode
        self.error = None
        layers.painting.editable = False
        self._active.add(self)

    @Slot()
    def finished(self):
        if self.predict:
            try:
                self.layers.refresh_predictions()
            except Exception as error:
                self.error = str(error)
        try:
            self.layers.painting.editable = self.editable
            if self.editable:
                self.layers.painting.mode = self.mode
        except Exception as error:
            self.error = str(error)
        finally:
            self._active.discard(self)


class LocalSegmentationControls(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Local Segmentation", parent)
        layout = QVBoxLayout(self)
        self.dataset_label = QLabel("No local dataset attached.")
        self.status_label = QLabel("Attach a local dataset to enable training.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.dataset_label)
        buttons = QHBoxLayout()
        self.train_button = QPushButton("Train")
        self.predict_button = QPushButton("Predict and Save")
        buttons.addWidget(self.train_button)
        buttons.addWidget(self.predict_button)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)
        self.train_button.clicked.connect(self.train)
        self.predict_button.clicked.connect(self.predict)
        self.dataset = self.layers = self.manager = None
        self.busy = self.trained = False
        self._worker = self._guard = None
        self._operation = None
        self._update_buttons()

    def set_dataset(self, dataset, layers, manager, *, name="CellCanvas", trained=False):
        """Attach a caller-owned core manager; serialize calls when sharing it."""
        if self.busy:
            raise RuntimeError("Wait for the current local operation to finish.")
        self.dataset, self.layers, self.manager = dataset, layers, manager
        self.trained = trained
        self.dataset_label.setText(name)
        self.status_label.setText("Ready to predict." if trained else "Paint labels, then train.")
        self._update_buttons()

    def _update_buttons(self):
        self.train_button.setEnabled(callable(getattr(self.manager, "fit", None)) and not self.busy)
        self.predict_button.setEnabled(callable(getattr(self.manager, "predict_to_segmentation", None))
                                       and self.trained and not self.busy)

    @Slot()
    def train(self):
        if callable(getattr(self.manager, "fit", None)) and not self.busy:
            self._start("Training", self.manager.fit)

    @Slot()
    def predict(self):
        if callable(getattr(self.manager, "predict_to_segmentation", None)) and self.trained and not self.busy:
            self._start("Prediction", lambda manager=self.manager, dataset=self.dataset:
                        manager.predict_to_segmentation(dataset))

    def _start(self, operation, work):
        self.busy = True
        self._operation = operation
        self._update_buttons()
        self.status_label.setText(f"{operation}...")
        self._guard = _LayerGuard(self.layers, operation == "Prediction")
        try:
            worker = create_worker(work, _start_thread=False, _ignore_errors=True)
            worker._layer_guard = self._guard
            worker.returned.connect(self._succeeded, Qt.QueuedConnection)
            worker.errored.connect(self._failed, Qt.QueuedConnection)
            worker.finished.connect(self._guard.finished, Qt.QueuedConnection)
            worker.finished.connect(self._finished, Qt.QueuedConnection)
            self._worker = worker
            worker.start()
        except Exception as error:
            self._guard.finished()
            self._failed(error)
            self._finished()

    @Slot(object)
    def _succeeded(self, result):
        if self._operation == "Training":
            self.trained = True
            self.status_label.setText("Training finished.")
        else:
            self.status_label.setText("Predictions saved.")

    @Slot(object)
    def _failed(self, error):
        if self._operation == "Training":
            self.trained = False
        self.status_label.setText(f"{self._operation} failed: {error}")

    @Slot()
    def _finished(self):
        if self._guard.error:
            self.status_label.setText(f"Operation finished; display update failed: {self._guard.error}")
        self.busy = False
        self._worker = None
        self._update_buttons()
