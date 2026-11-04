"""Plain HTTP adapter for the core service; no viewer imports."""

import ipaddress
import time
from urllib.parse import quote, urlsplit
from uuid import uuid4

import requests


class RemoteUncertain(RuntimeError):
    """The server may still own the dataset; painting must stay locked."""


class CoreServiceClient:
    def __init__(self, url, *, timeout=(1, 5)):
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Enter an HTTP service URL.")
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            local = parsed.hostname == "localhost"
        if not local:
            raise ValueError("This connection currently requires a service on this machine and shared local Zarr stores.")
        self.url, self.timeout = url.rstrip("/"), timeout

    def request(self, method, path, body=None):
        response = requests.request(method, self.url + path, json=body, timeout=self.timeout)
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            error = RuntimeError(f"Service error {response.status_code}: {detail}")
            error.status = response.status_code
            raise error
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("The service returned an invalid response.")
        return data

    def datasets(self):
        datasets = self.request("GET", "/datasets").get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError("The service did not return registered datasets.")
        return datasets


def local_storage(dataset):
    from zarr.storage import LocalStore

    def location(array):
        if not isinstance(getattr(array, "store", None), LocalStore):
            raise ValueError("Service attachment currently requires shared local Zarr stores.")
        return {"uri": array.store.root.resolve().as_uri(), "path": array.path}

    return {"image": location(dataset.image), "painting": location(dataset.labels),
            "prediction": location(dataset.segmentation),
            "features": [location(array) for array in dataset.features.values()]}


class CoreServiceManager:
    def __init__(self, dataset, url, name, *, timeout=(1, 5), wait_timeout=30, poll_interval=.1):
        if not name:
            raise ValueError("Select a registered dataset.")
        self.dataset, self.name = dataset, name
        self.client = CoreServiceClient(url, timeout=timeout)
        self.path = "/datasets/" + quote(name, safe="")
        self.storage = local_storage(dataset)
        self.wait_timeout, self.poll_interval = wait_timeout, poll_interval
        self.service_id = None
        self.trained = False
        self.safe_to_edit = True
        self._pending = None

    def _metadata(self):
        metadata = self.client.request("GET", self.path)
        geometry = self.dataset.spatial_transform
        feature_axes = ["z", "y", "x"]
        feature_axes.insert(self.dataset.feature_axis % 4, "d")
        expected = {
            "image": {"axes": ["c", "z", "y", "x"], "shape": list(self.dataset.image.shape)},
            "painting": {"axes": ["z", "y", "x"], "shape": list(self.dataset.labels.shape)},
            "features": {"axis": self.dataset.feature_axis, "axes": feature_axes,
                         "shapes": [list(array.shape) for array in self.dataset.features.values()]},
            "spatial_transform": {"scale": list(geometry.scale), "translation": list(geometry.translation),
                                  "units": list(geometry.units)},
            "storage": self.storage,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("The registered dataset does not match the attached stores, axes or geometry.")
        prediction = metadata.get("prediction", {})
        if (prediction.get("axes") != ["z", "y", "x"]
                or prediction.get("shape") != list(self.dataset.segmentation.shape)
                or not isinstance(metadata.get("busy"), bool)
                or not isinstance(metadata.get("trained"), bool)
                or not isinstance(metadata.get("service_id"), str)):
            raise ValueError("The service returned incomplete dataset metadata.")
        self.trained = metadata["trained"]
        return metadata

    def connect(self):
        metadata = self._metadata()
        self.service_id = metadata["service_id"]
        self.safe_to_edit = not metadata["busy"]
        if not self.safe_to_edit:
            raise RemoteUncertain("The service is busy. Painting stays locked; check service status.")
        return metadata

    def fit(self):
        return self._operate("train", {})

    def predict_to_segmentation(self, dataset, batch_size=65536):
        if dataset is not self.dataset:
            raise ValueError("This service manager is bound to another dataset.")
        return self._operate("predict", {"batch_size": batch_size})

    def _operate(self, operation, options):
        if not self.safe_to_edit:
            raise RemoteUncertain("Check the pending service operation before starting another.")
        metadata = self._metadata()
        self.service_id = metadata["service_id"]
        self._pending = (operation, dict(options, operation_id=uuid4().hex, service_id=self.service_id))
        self.safe_to_edit = False
        return self.reconcile()

    def reconcile(self):
        deadline = time.monotonic() + self.wait_timeout
        try:
            while True:
                metadata = self._metadata()
                if metadata["service_id"] != self.service_id:
                    self.safe_to_edit = not metadata["busy"]
                    if self.safe_to_edit:
                        self._pending = None
                        self.service_id = metadata["service_id"]
                        raise RuntimeError("Service restarted; previous work was not confirmed. Train again.")
                    raise RemoteUncertain("Service restarted and is busy; painting stays locked.")
                if self._pending is None:
                    if not metadata["busy"]:
                        self.safe_to_edit = True
                        return metadata
                else:
                    operation, body = self._pending
                    try:
                        receipt = self.client.request("GET", self.path + "/operations/" + body["operation_id"])
                    except RuntimeError as error:
                        if getattr(error, "status", None) != 404:
                            raise
                        # Safe even when a previous POST reply was lost: the
                        # service reserves this ID and will not execute it twice.
                        receipt = self.client.request("POST", self.path + "/" + operation, body)
                    if (receipt.get("id") != body["operation_id"]
                            or receipt.get("service_id") != self.service_id
                            or receipt.get("operation") != operation
                            or receipt.get("state") not in ("running", "succeeded", "failed")):
                        raise ValueError("Invalid service operation receipt.")
                    if receipt["state"] != "running" and not metadata["busy"]:
                        # Re-read after the receipt: metadata could predate
                        # submission or completion of the operation.
                        metadata = self._metadata()
                        if not metadata["busy"]:
                            self.safe_to_edit = True
                            self._pending = None
                            if receipt["state"] == "failed":
                                raise RuntimeError(receipt.get("error", {}).get("message", "Core operation failed."))
                            if operation == "predict" and metadata["prediction"].get("status") != "ready":
                                raise RuntimeError("The latest prediction is not ready.")
                            return receipt["result"]
                if time.monotonic() >= deadline:
                    raise RemoteUncertain("The operation is still unconfirmed. Painting stays locked; check service status.")
                time.sleep(self.poll_interval)
        except Exception as error:
            if not self.safe_to_edit and not isinstance(error, RemoteUncertain):
                raise RemoteUncertain(f"{error}. Painting stays locked; check service status.") from error
            raise
