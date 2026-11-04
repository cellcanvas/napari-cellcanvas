"""Select an explicitly named core-service dataset."""

from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton
from superqt.utils import create_worker

from ._core_client import CoreServiceClient


class ServiceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Core Service")
        layout = QFormLayout(self)
        self.url = QLineEdit("http://127.0.0.1:8001")
        self.load_button = QPushButton("List Registered Datasets")
        self.datasets = QComboBox()
        self.status = QLabel("Use the same local Zarr stores as the service.")
        self.status.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow("Service URL", self.url)
        layout.addRow(self.load_button)
        layout.addRow("Dataset", self.datasets)
        layout.addRow(self.status)
        layout.addRow(self.buttons)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.load_button.clicked.connect(self.load)
        self.datasets.currentIndexChanged.connect(self._selected)
        self.url.textChanged.connect(self._changed_url)
        self._worker = None

    def _changed_url(self):
        self.datasets.clear()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def _selected(self):
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(self.datasets.currentData() is not None)

    @Slot()
    def load(self):
        try:
            client = CoreServiceClient(self.url.text())
            self.url.setEnabled(False)
            self.load_button.setEnabled(False)
            self.datasets.clear()
            self.status.setText("Loading datasets...")
            self._worker = create_worker(client.datasets, _start_thread=False, _ignore_errors=True)
            self._worker.returned.connect(self._loaded, Qt.QueuedConnection)
            self._worker.errored.connect(self._failed, Qt.QueuedConnection)
            self._worker.finished.connect(self._finished, Qt.QueuedConnection)
            self._worker.start()
        except Exception as error:
            self._failed(error)
            self._finished()

    @Slot(object)
    def _loaded(self, datasets):
        self.datasets.addItem("Select a dataset…", None)
        for name in datasets:
            self.datasets.addItem(name, name)
        self.status.setText("Select the dataset registered for your attached stores.")

    @Slot(object)
    def _failed(self, error):
        self.status.setText(str(error))

    @Slot()
    def _finished(self):
        self._worker = None
        self.url.setEnabled(True)
        self.load_button.setEnabled(True)
