import dask.array as da
import numpy as np
import copick
import zarr
import napari
import sys
import requests
from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QLineEdit,
    QFormLayout,
)
from qtpy.QtCore import Qt
from napari.utils import DirectLabelColormap
from superqt.utils import thread_worker
import logging

class CopickPlugin(QWidget):
    def __init__(self, viewer=None, config_path=None):
        super().__init__()
        if viewer:
            self.viewer = viewer
        else:
            self.viewer = napari.Viewer()

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.root = None
        self.selected_run_id = None
        self.current_layer = None
        self.session_id = "0"
        self.setup_ui()
        if config_path:
            self.load_config(config_path)

    def setup_ui(self):
        layout = QVBoxLayout()

        # FastAPI URL input (prepopulated)
        self.api_url_input = QLineEdit(self)
        self.api_url_input.setText("http://localhost:8000")  # Prepopulate with default URL
        self.api_url_input.setPlaceholderText("FastAPI URL")
        layout.addWidget(self.api_url_input)

        # Load Config button
        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.load_button)

        # Run ID selection dropdown (previously Dataset)
        self.run_id_dropdown = QComboBox()
        layout.addWidget(self.run_id_dropdown)

        # Tomogram Type input
        self.tomo_type_input = QLineEdit(self)
        self.tomo_type_input.setText("wbp")
        self.tomo_type_input.setPlaceholderText("Tomo Type")
        layout.addWidget(self.tomo_type_input)

        # Annotation Name input
        self.annotation_input = QLineEdit(self)
        self.annotation_input.setText("cellcanvasAnnotation")
        self.annotation_input.setPlaceholderText("Annotation Name")
        layout.addWidget(self.annotation_input)

        # Prediction Name input
        self.prediction_input = QLineEdit(self)
        self.prediction_input.setText("cellcanvasPrediction")
        self.prediction_input.setPlaceholderText("Prediction Name")
        layout.addWidget(self.prediction_input)

        # Model Name input
        self.model_input = QLineEdit(self)
        self.model_input.setText("/mnt/cellcanvasModel.joblib")
        self.model_input.setPlaceholderText("Model Name")
        layout.addWidget(self.model_input)

        # User ID input (default: cellcanvasUser)
        self.user_id_input = QLineEdit(self)
        self.user_id_input.setText("napariCopick")
        self.user_id_input.setPlaceholderText("User ID")
        layout.addWidget(self.user_id_input)

        # Feature Type input (for generating features)
        self.feature_type_input = QLineEdit(self)
        self.feature_type_input.setText("multiscaleTorchFeatures")
        self.feature_type_input.setPlaceholderText("Feature Type")
        layout.addWidget(self.feature_type_input)

        # Open Tomogram button
        self.open_tomogram_button = QPushButton("Open Tomogram")
        self.open_tomogram_button.clicked.connect(self.open_tomogram)
        layout.addWidget(self.open_tomogram_button)

        # Open Annotation Segmentation button
        self.open_annotation_button = QPushButton("Open Annotation Segmentation")
        self.open_annotation_button.clicked.connect(self.open_annotation)
        layout.addWidget(self.open_annotation_button)

        # Open Prediction Segmentation button
        self.open_prediction_button = QPushButton("Open Prediction Segmentation")
        self.open_prediction_button.clicked.connect(self.open_prediction)
        layout.addWidget(self.open_prediction_button)

        # Generate Features button
        self.generate_features_button = QPushButton("Generate Features")
        self.generate_features_button.clicked.connect(self.call_generate_features)
        layout.addWidget(self.generate_features_button)

        # Train Model button
        self.train_model_button = QPushButton("Train Model")
        self.train_model_button.clicked.connect(self.call_train_model)
        layout.addWidget(self.train_model_button)

        # Run Model button
        self.run_model_button = QPushButton("Run Model")
        self.run_model_button.clicked.connect(self.call_run_model)
        layout.addWidget(self.run_model_button)

        # Train on All Data button
        self.train_all_button = QPushButton("Train on All Data")
        self.train_all_button.clicked.connect(self.call_train_all_data)
        layout.addWidget(self.train_all_button)

        # Predict on All Data button
        self.predict_all_button = QPushButton("Predict on All Data")
        self.predict_all_button.clicked.connect(self.call_predict_all_data)
        layout.addWidget(self.predict_all_button)        

        # Status Label
        self.status_label = QLabel("Status: Not fetched yet")
        layout.addWidget(self.status_label)

        # Update Status button
        self.update_status_button = QPushButton("Update Status")
        self.update_status_button.clicked.connect(self.update_status)
        layout.addWidget(self.update_status_button)

        self.setLayout(layout)

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Config", "", "JSON Files (*.json)"
        )
        if path:
            self.load_config(path)

    def load_config(self, path=None):
        if path:
            self.root = copick.from_file(path)
            self.populate_dropdown()

    def populate_dropdown(self):
        self.run_id_dropdown.clear()
        for run in self.root.runs:
            self.run_id_dropdown.addItem(run.meta.name)

    def get_api_url(self):
        return self.api_url_input.text()  # Use the text from the input field

    def get_voxel_spacing(self):
        selected_run_id = self.get_selected_run_id()
        run = self.get_run(selected_run_id)
        if run and run.voxel_spacings:
            return run.voxel_spacings[0].meta.voxel_size  # Automatically use the first voxel spacing
        else:
            print("Error: Voxel spacing not found.")
            return None

    @thread_worker
    def generate_features_worker(self, data):
        response = requests.post(f"{self.get_api_url()}/generate-features", json=data)
        return response

    @thread_worker
    def train_model_worker(self, data):
        response = requests.post(f"{self.get_api_url()}/train-model", json=data)
        return response

    @thread_worker
    def run_model_worker(self, data):
        response = requests.post(f"{self.get_api_url()}/run-model", json=data)
        return response

    def call_generate_features(self):
        selected_run_id = self.get_selected_run_id()
        voxel_spacing = self.get_voxel_spacing()
        tomo_type = self.tomo_type_input.text()
        feature_type = self.feature_type_input.text()

        if voxel_spacing is None:
            self.logger.error("Voxel spacing not found. Aborting generate_features.")
            return

        data = {
            "copick_config_path": "auto-detect",
            "run_name": selected_run_id,
            "voxel_spacing": float(voxel_spacing),  # Ensure this is a float
            "tomo_type": tomo_type,
            "feature_type": feature_type,
            "intensity": True,
            "edges": True,
            "texture": True,
            "sigma_min": 0.5,
            "sigma_max": 4.0
        }

        self.logger.info(f"Sending generate_features request to server: {data}")

        try:
            response = requests.post(f"{self.get_api_url()}/generate-features", json=data)
            self.logger.info(f"Generate Features Response: {response.status_code} - {response.text}")
            self.handle_generate_features_response(response)
        except requests.RequestException as e:
            self.logger.error(f"Error sending generate_features request: {str(e)}")

    def handle_generate_features_response(self, response):
        self.logger.info(f"Generate Features Response: {response.status_code} - {response.text}")

    def call_train_model(self):
        selected_run_id = self.get_selected_run_id()
        voxel_spacing = self.get_voxel_spacing()
        session_id = self.session_id
        user_id = self.user_id_input.text()
        painting_segmentation_names = self.annotation_input.text()

        if voxel_spacing is None:
            self.logger.error("Voxel spacing not found. Aborting train_model.")
            return

        feature_types = self.feature_type_input.text()

        data = {
            "copick_config_path": "auto-detect",
            "painting_segmentation_names": painting_segmentation_names,
            "session_id": session_id,
            "user_id": user_id,
            "voxel_spacing": float(voxel_spacing),
            "tomo_type": self.tomo_type_input.text(),
            "feature_types": feature_types,
            "run_names": selected_run_id,
            "eta": 0.3,
            "gamma": 0.0,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "max_delta_step": 0.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "max_bin": 256,
            "output_model_path": f"{self.model_input.text()}"
        }

        self.logger.info(f"Sending train_model request to server: {data}")

        try:
            response = requests.post(f"{self.get_api_url()}/train-model", json=data)
            self.logger.info(f"Train Model Response: {response.status_code} - {response.text}")
            self.handle_train_model_response(response)
        except requests.RequestException as e:
            self.logger.error(f"Error sending train_model request: {str(e)}")

    def handle_train_model_response(self, response):
        self.logger.info(f"Train Model Response: {response.status_code} - {response.text}")

    def call_run_model(self):
        selected_run_id = self.get_selected_run_id()
        voxel_spacing = self.get_voxel_spacing()
        session_id = self.session_id
        user_id = self.user_id_input.text()
        model_path = f"{self.model_input.text()}"
        feature_names = self.feature_type_input.text()
        segmentation_name = self.prediction_input.text()

        if voxel_spacing is None:
            return

        data = {
            "copick_config_path": self.get_api_url(),
            "session_id": session_id,
            "user_id": user_id,
            "voxel_spacing": voxel_spacing,
            "run_name": selected_run_id,
            "model_path": model_path,
            "tomo_type": self.tomo_type_input.text(),
            "feature_names": feature_names,
            "segmentation_name": segmentation_name
        }

        self.logger.info(f"Sending run_model request to server: {data}")

        try:
            response = requests.post(f"{self.get_api_url()}/run-model", json=data)
            self.logger.info(f"Run Model Response: {response.status_code} - {response.text}")
            self.handle_run_model_response(response)
        except requests.RequestException as e:
            self.logger.error(f"Error sending run_model request: {str(e)}")

    def handle_run_model_response(self, response):
        self.logger.info(f"Run Model Response: {response.status_code} - {response.text}")

    def call_train_all_data(self):
        """Send request to train model on all available data (no specific run)"""
        voxel_spacing = self.get_voxel_spacing()
        session_id = self.session_id
        user_id = self.user_id_input.text()
        painting_segmentation_names = self.annotation_input.text()

        if voxel_spacing is None:
            self.logger.error("Voxel spacing not found. Aborting train_model on all data.")
            return

        feature_types = self.feature_type_input.text()

        data = {
            "copick_config_path": "auto-detect",
            "painting_segmentation_names": painting_segmentation_names,
            "session_id": session_id,
            "user_id": user_id,
            "run_names": "all",
            "voxel_spacing": float(voxel_spacing),
            "tomo_type": self.tomo_type_input.text(),
            "feature_types": feature_types,
            "eta": 0.3,
            "gamma": 0.0,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "max_delta_step": 0.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "max_bin": 256,
            "output_model_path": f"{self.model_input.text()}"
        }

        self.logger.info(f"Sending train_all_data request to server: {data}")

        try:
            response = requests.post(f"{self.get_api_url()}/train-all", json=data)
            self.logger.info(f"Train on All Data Response: {response.status_code} - {response.text}")
            self.handle_train_model_response(response)
        except requests.RequestException as e:
            self.logger.error(f"Error sending train_all_data request: {str(e)}")

    def call_predict_all_data(self):
        """Send request to predict on all available data (no specific run)"""
        voxel_spacing = self.get_voxel_spacing()
        session_id = self.session_id
        user_id = self.user_id_input.text()
        model_path = f"{self.model_input.text()}"
        feature_names = self.feature_type_input.text()
        segmentation_name = self.prediction_input.text()

        if voxel_spacing is None:
            self.logger.error("Voxel spacing not found. Aborting predict_all_data.")
            return

        data = {
            "copick_config_path": self.get_api_url(),
            "session_id": session_id,
            "user_id": user_id,
            "voxel_spacing": voxel_spacing,
            "model_path": model_path,
            "tomo_type": self.tomo_type_input.text(),
            "feature_names": feature_names,
            "segmentation_name": segmentation_name
        }

        self.logger.info(f"Sending predict_all_data request to server: {data}")

        try:
            response = requests.post(f"{self.get_api_url()}/predict-all", json=data)
            self.logger.info(f"Predict on All Data Response: {response.status_code} - {response.text}")
            self.handle_run_model_response(response)
        except requests.RequestException as e:
            self.logger.error(f"Error sending predict_all_data request: {str(e)}")

    def open_tomogram(self):
        selected_run_id = self.get_selected_run_id()
        tomo_type = self.tomo_type_input.text()

        run = self.get_run(selected_run_id)
        if run:
            voxel_spacing = run.voxel_spacings[0]
            for tomogram in voxel_spacing.tomograms:
                if tomogram.meta.tomo_type == tomo_type:
                    self.load_tomogram(tomogram)
                    return
        print("Tomogram not found")

    def open_annotation(self):
        selected_run_id = self.get_selected_run_id()
        annotation_name = self.annotation_input.text()

        run = self.get_run(selected_run_id)
        if run:
            voxel_spacing = run.voxel_spacings[0]
            segmentations = voxel_spacing.run.get_segmentations(
                voxel_size=voxel_spacing.meta.voxel_size
            )
            for segmentation in segmentations:
                if segmentation.meta.name == annotation_name:
                    self.load_segmentation(segmentation)
                    return
        print("Annotation not found, creating new annotation")
        self.create_segmentation(run, annotation_name)

    def open_prediction(self):
        selected_run_id = self.get_selected_run_id()
        prediction_name = self.prediction_input.text()

        run = self.get_run(selected_run_id)
        if run:
            voxel_spacing = run.voxel_spacings[0]
            segmentations = voxel_spacing.run.get_segmentations(
                voxel_size=voxel_spacing.meta.voxel_size
            )
            for segmentation in segmentations:
                if segmentation.meta.name == prediction_name:
                    self.load_segmentation(segmentation)
                    return
        print("Prediction not found, creating new prediction")
        self.create_segmentation(run, prediction_name)

    def get_selected_run_id(self):
        return self.run_id_dropdown.currentText()

    def get_run(self, name):
        return self.root.get_run(name)

    def load_tomogram(self, tomogram):
        zarr_path = zarr.storage.LRUStoreCache(tomogram.zarr(), max_size=2**32)
        zarr_group = zarr.open(zarr_path, "r")

        scale_levels = [key for key in zarr_group.keys() if key.isdigit()]
        scale_levels.sort(key=int)

        self.viewer.add_image([zarr_group[scale] for scale in scale_levels])
        # self.info_label.setText(f"Loaded Tomogram: {tomogram.meta.tomo_type}")

    def load_segmentation(self, segmentation):
        zarr_data = zarr.open(store=segmentation.zarr(), mode="a", path="/")
        data = zarr_data["0"]
        scale = [1, 1, 1]

        colormap = self.get_copick_colormap()
        painting_layer = self.viewer.add_labels(
            data, name=f"Segmentation: {segmentation.meta.name}", scale=scale
        )
        # painting_layer.colormap = DirectLabelColormap(color_dict=colormap)
        painting_layer.painting_labels = [
            obj.label for obj in self.root.config.pickable_objects
        ]

    def create_segmentation(self, run, name):
        seg = run.new_segmentation(
            voxel_size=run.voxel_spacings[0].meta.voxel_size,
            name=name,
            session_id="0",
            is_multilabel=True,
            user_id="napariCopick",
        )

        tomo = zarr.open(run.voxel_spacings[0].tomograms[0].zarr(), "r")["0"]

        zarr_file = zarr.open(store=seg.zarr(), mode="a", path="/")
        zarr_file.create_dataset(
            "0", 
            shape=tomo.shape, 
            dtype=np.int8, 
            chunks=(128, 128, 128), 
            fill_value=0, 
            dimension_separator="/"
        )

        self.load_segmentation(seg)

    def get_copick_colormap(self, pickable_objects=None):
        if not pickable_objects:
            pickable_objects = self.root.config.pickable_objects
        colormap = {
            obj.label: np.array(obj.color) / 255.0 for obj in pickable_objects
        }
        colormap[None] = np.array([1, 1, 1, 1])
        return colormap

    def update_status(self):
        selected_run_id = self.get_selected_run_id()
        url = f"{self.get_api_url()}/status?dataset={selected_run_id}"
        self.logger.info(f"Fetching status from: {url}")
        
        response = requests.get(url)
        if response.status_code == 200:
            status_data = response.json()
            self.logger.info(f"Received status data: {status_data}")

            # Use .get() to safely retrieve the value or provide a default value
            run_id = status_data.get('run_id', 'Unknown')
            num_runs = status_data.get('runs', 0)
            features_exist = 'Yes' if status_data.get('features_exist', False) else 'No'
            annotations_exist = 'Yes' if status_data.get('annotations_exist', False) else 'No'
            predictions_exist = 'Yes' if status_data.get('predictions_exist', False) else 'No'
            feature_generation_status = status_data.get('feature_generation', 'Unknown')
            model_training_status = status_data.get('model_training', 'Unknown')
            model_inference_status = status_data.get('model_inference', 'Unknown')

            # Construct the status text
            status_text = (
                f"<b>Run ID:</b> {run_id}<br>"
                f"<b>Number of Runs:</b> {num_runs}<br>"
                f"<b>Features Exist:</b> {features_exist}<br>"
                f"<b>Annotations Exist:</b> {annotations_exist}<br>"
                f"<b>Predictions Exist:</b> {predictions_exist}<br><br>"
                f"<b>Feature Generation Status:</b> {feature_generation_status}<br>"
                f"<b>Model Training Status:</b> {model_training_status}<br>"
                f"<b>Model Inference Status:</b> {model_inference_status}<br>"
            )
            self.status_label.setText(status_text)
            self.status_label.setTextFormat(Qt.RichText)
        else:
            self.logger.error(f"Error fetching status: {response.status_code} - {response.text}")
            self.status_label.setText(f"Error fetching status: {response.status_code}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copick Plugin")
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Path to the copick config file",
    )
    args = parser.parse_args()

    config_path = args.config_path if args.config_path else None
    config_path = "/Users/kharrington/Data/copick/CZCDP_10048_local.json"

    viewer = napari.Viewer()
    copick_plugin = CopickPlugin(viewer, config_path=config_path)
    viewer.window.add_dock_widget(copick_plugin, area="right")
    napari.run()
