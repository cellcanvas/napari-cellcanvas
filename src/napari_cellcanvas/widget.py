import napari
import numpy as np
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, 
                            QLineEdit, QFormLayout, QScrollArea, QTreeWidget, QTreeWidgetItem,
                            QHBoxLayout, QMenu, QAction, QSpinBox, QCheckBox, QListWidget,
                            QListWidgetItem)
from qtpy.QtCore import Qt
import requests
import copick
import zarr
from napari.utils import DirectLabelColormap

class MultiSelectComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Ensure scroll bar is always visible
        self.setEditable(False)  # Disable text editing in the combo box

        # Create the list widget with checkable items
        self.list_widget = QListWidget()
        self.setView(self.list_widget)

        # Create a button for clearing selections (optional)
        self.clear_button = QPushButton("Clear Selections", self)
        self.clear_button.clicked.connect(self.clear_selections)
        self.view().setItemWidget(QListWidgetItem(), self.clear_button)

    def addItems(self, items):
        for item in items:
            list_item = QListWidgetItem(item)
            list_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            list_item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(list_item)

    def selectedItems(self):
        selected = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def currentText(self):
        return ", ".join(self.selectedItems())

    def clear_selections(self):
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            item.setCheckState(Qt.Unchecked)

class CellCanvasWidget(QWidget):
    def __init__(self, viewer=None, copick_config_path="/Volumes/CZII_A/cellcanvas_tutorial/new/cellcanvas_tutorial/dataset_10229_copick.json", hostname="localhost", port=8000, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("CellCanvas Widget")
        self.hostname = hostname
        self.port = port
        self.copick_config_path = copick_config_path
        self.layout = QVBoxLayout(self)

        print(f"Checking server connection at http://{self.hostname}:{self.port}")
        try:
            response = requests.get(f"http://{self.hostname}:{self.port}/index")
            print(f"Server response status: {response.status_code}")
            print(f"Server response content: {response.text[:100]}...")  # Print first 100 characters
        except Exception as e:
            print(f"Error connecting to server: {e}")
        
        # Load Copick project
        self.root = copick.from_file(self.copick_config_path)
        
        # Add refresh button
        self.refresh_button = QPushButton("Refresh", self)
        self.refresh_button.clicked.connect(self.refresh_tree)
        self.layout.addWidget(self.refresh_button)
        
        # Hierarchical tree view
        self.tree_view = QTreeWidget()
        self.tree_view.setHeaderLabel("Copick Project")
        self.tree_view.itemExpanded.connect(self.handle_item_expand)
        self.tree_view.itemClicked.connect(self.handle_item_click)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(
            self.open_context_menu
        )
        self.layout.addWidget(self.tree_view)

        # Run selection dropdown
        self.run_dropdown = QComboBox(self)
        self.layout.addWidget(QLabel("Select Run:"))
        self.layout.addWidget(self.run_dropdown)
        self.populate_run_dropdown()
        
        # Solution selection dropdown
        self.solution_dropdown = QComboBox(self)
        self.layout.addWidget(QLabel("Select Solution:"))
        self.layout.addWidget(self.solution_dropdown)
        self.populate_solution_dropdown()
        
        # Scroll area for solution arguments
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QFormLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll_area)
        
        # Run solution button
        self.run_button = QPushButton("Run Solution", self)
        self.run_button.clicked.connect(self.run_solution)
        self.layout.addWidget(self.run_button)
        
        self.setLayout(self.layout)
        self.solution_dropdown.currentIndexChanged.connect(self.update_solution_args)
        self.populate_tree()

    def populate_run_dropdown(self):
        self.run_dropdown.clear()
        for run in self.root.runs:
            self.run_dropdown.addItem(run.meta.name)        

    def populate_tree(self):
        self.tree_view.clear()
        for run in self.root.runs:
            run_item = QTreeWidgetItem(self.tree_view, [run.meta.name])
            run_item.setData(0, Qt.UserRole, run)
            run_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)

    def handle_item_expand(self, item):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            self.expand_run(item, data)
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            self.expand_voxel_spacing(item, data)

    def expand_run(self, item, run):
        if not item.childCount():
            for voxel_spacing in run.voxel_spacings:
                spacing_item = QTreeWidgetItem(
                    item, [f"Voxel Spacing: {voxel_spacing.meta.voxel_size}"]
                )
                spacing_item.setData(0, Qt.UserRole, voxel_spacing)
                spacing_item.setChildIndicatorPolicy(
                    QTreeWidgetItem.ShowIndicator
                )

            # Add picks nested by user_id, session_id, and pickable_object_name
            picks = run.picks
            picks_item = QTreeWidgetItem(item, ["Picks"])
            user_dict = {}
            for pick in picks:
                if pick.meta.user_id not in user_dict:
                    user_dict[pick.meta.user_id] = {}
                if pick.meta.session_id not in user_dict[pick.meta.user_id]:
                    user_dict[pick.meta.user_id][pick.meta.session_id] = []
                user_dict[pick.meta.user_id][pick.meta.session_id].append(pick)

            for user_id, sessions in user_dict.items():
                user_item = QTreeWidgetItem(picks_item, [f"User: {user_id}"])
                for session_id, picks in sessions.items():
                    session_item = QTreeWidgetItem(
                        user_item, [f"Session: {session_id}"]
                    )
                    for pick in picks:
                        pick_child = QTreeWidgetItem(
                            session_item, [pick.meta.pickable_object_name]
                        )
                        pick_child.setData(0, Qt.UserRole, pick)
            item.addChild(picks_item)

    def expand_voxel_spacing(self, item, voxel_spacing):
        if not item.childCount():
            tomogram_item = QTreeWidgetItem(item, ["Tomograms"])
            for tomogram in voxel_spacing.tomograms:
                tomo_child = QTreeWidgetItem(
                    tomogram_item, [tomogram.meta.tomo_type]
                )
                tomo_child.setData(0, Qt.UserRole, tomogram)
            item.addChild(tomogram_item)

            segmentation_item = QTreeWidgetItem(item, ["Segmentations"])
            segmentations = voxel_spacing.run.get_segmentations(
                voxel_size=voxel_spacing.meta.voxel_size
            )
            for segmentation in segmentations:
                seg_child = QTreeWidgetItem(
                    segmentation_item, [segmentation.meta.name]
                )
                seg_child.setData(0, Qt.UserRole, segmentation)
            item.addChild(segmentation_item)

            # Add Features item
            features_item = QTreeWidgetItem(item, ["Features"])
            for tomogram in voxel_spacing.tomograms:
                for feature in tomogram.features:
                    feature_child = QTreeWidgetItem(
                        features_item, [f"{feature.feature_type} ({tomogram.meta.tomo_type})"]
                    )
                    feature_child.setData(0, Qt.UserRole, feature)
            item.addChild(features_item)

    def handle_item_click(self, item, column):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, copick.models.CopickRun):
            # self.info_label.setText(f"Run: {data.meta.name}")
            self.selected_run = data
        elif isinstance(data, copick.models.CopickVoxelSpacing):
            # self.info_label.setText(f"Voxel Spacing: {data.meta.voxel_size}")
            self.lazy_load_voxel_spacing(item, data)
        elif isinstance(data, copick.models.CopickTomogram):
            self.load_tomogram(data)
        elif isinstance(data, copick.models.CopickSegmentation):
            self.load_segmentation(data)
        elif isinstance(data, copick.models.CopickPicks):
            parent_run = self.get_parent_run(item)
            self.load_picks(data, parent_run)

    def get_parent_run(self, item):
        while item:
            data = item.data(0, Qt.UserRole)
            if isinstance(data, copick.models.CopickRun):
                return data
            item = item.parent()
        return None

    def lazy_load_voxel_spacing(self, item, voxel_spacing):
        if not item.childCount():
            self.expand_voxel_spacing(item, voxel_spacing)

    def load_tomogram(self, tomogram):
        zarr_path = tomogram.zarr()
        zarr_group = zarr.open(zarr_path, "r")

        # Determine the number of scale levels
        scale_levels = [key for key in zarr_group.keys() if key.isdigit()]
        scale_levels.sort(key=int)

        data = [zarr_group[level] for level in scale_levels]

        # data = [da.from_zarr(str(zarr_path), level) * (int(level) + 1) / 2 for level in scale_levels]

        # Highest scale level only
        # data = zarr.open(tomogram.zarr(), 'r')["0"]

        # TODO scale needs to account for scale pyramid (4x for the lowest scale in this case)
        
        scale = [tomogram.voxel_spacing.meta.voxel_size] * 3
        self.viewer.add_image(
            data, name=f"Tomogram: {tomogram.meta.tomo_type}", scale=scale
        )
        # self.info_label.setText(
        #     f"Loaded Tomogram: {tomogram.meta.tomo_type} with num scales = {len(scale_levels)}"
        # )

    def load_segmentation(self, segmentation):
        zarr_data = zarr.open(segmentation.zarr(), "a")
        if "data" in zarr_data:
            data = zarr_data["data"]
        else:
            data = zarr_data[:]

        scale = [segmentation.meta.voxel_size] * 3

        # Create a color map based on copick colors
        colormap = self.get_copick_colormap()
        painting_layer = self.viewer.add_labels(
            data, name=f"Segmentation: {segmentation.meta.name}", scale=scale
        )
        painting_layer.colormap = DirectLabelColormap(color_dict=colormap)
        painting_layer.painting_labels = [
            obj.label for obj in self.root.config.pickable_objects
        ]
        self.class_labels_mapping = {
            obj.label: obj.name for obj in self.root.config.pickable_objects
        }

        # self.info_label.setText(
        #     f"Loaded Segmentation: {segmentation.meta.name}"
        # )

    def get_copick_colormap(self, pickable_objects=None):
        if not pickable_objects:
            pickable_objects = self.root.config.pickable_objects
        colormap = {
            obj.label: np.array(obj.color) / 255.0 for obj in pickable_objects
        }
        colormap[None] = np.array([1, 1, 1, 1])
        return colormap

    def load_picks(self, pick_set, parent_run):
        if parent_run is not None:
            if pick_set:
                if pick_set.points:
                    points = [
                        (p.location.z, p.location.y, p.location.x)
                        for p in pick_set.points
                    ]
                    color = (
                        pick_set.color
                        if pick_set.color
                        else (255, 255, 255, 255)
                    )  # Default to white if color is not set
                    colors = np.tile(
                        np.array(
                            [
                                color[0] / 255.0,
                                color[1] / 255.0,
                                color[2] / 255.0,
                                color[3] / 255.0,
                            ]
                        ),
                        (len(points), 1),
                    )  # Create an array with the correct shape
                    pickable_object = [
                        obj
                        for obj in self.root.pickable_objects
                        if obj.name == pick_set.pickable_object_name
                    ][0]
                    point_size = pickable_object.radius
                    self.viewer.add_points(
                        points,
                        name=f"Picks: {pick_set.meta.pickable_object_name}",
                        size=point_size,
                        face_color=colors,
                        out_of_slice_display=True,
                    )
                    # self.info_label.setText(
                    #     f"Loaded Picks: {pick_set.meta.pickable_object_name}"
                    # )
                # else:
                    # self.info_label.setText(
                    #     f"No points found for Picks: {pick_set.meta.pickable_object_name}"
                    # )
            # else:
                # self.info_label.setText(
                #     f"No pick set found for Picks: {pick_set.meta.pickable_object_name}"
                # )
        # else:
            # self.info_label.setText("No parent run found")

    def get_color(self, pick):
        for obj in self.root.pickable_objects:
            if obj.name == pick.meta.object_name:
                return obj.color
        return "white"

    def get_run(self, name):
        return self.root.get_run(name)

    def open_context_menu(self, position):
        print("Opening context menu")
        item = self.tree_view.itemAt(position)
        if not item:
            return

        if self.is_segmentations_or_picks_item(item):
            context_menu = QMenu(self.tree_view)
            if item.text(0) == "Segmentations":
                run_name = item.parent().parent().text(0)
                run = self.root.get_run(run_name)
                self.show_segmentation_widget(run)
            elif item.text(0) == "Picks":
                run_name = item.parent().text(0)
                run = self.root.get_run(run_name)
                self.show_picks_widget(run)
            context_menu.exec_(self.tree_view.viewport().mapToGlobal(position))

    def is_segmentations_or_picks_item(self, item):
        if item.text(0) == "Segmentations" or item.text(0) == "Picks":
            return True
        return False

    def show_segmentation_widget(self, run):
        widget = QWidget()
        widget.setWindowTitle("Create New Segmentation")

        layout = QFormLayout(widget)
        name_input = QLineEdit(widget)
        name_input.setText("segmentation")
        layout.addRow("Name:", name_input)

        session_input = QSpinBox(widget)
        session_input.setValue(0)
        layout.addRow("Session ID:", session_input)

        user_input = QLineEdit(widget)
        user_input.setText("napariCellcanvas")
        layout.addRow("User ID:", user_input)

        voxel_size_input = QComboBox(widget)
        for voxel_spacing in run.voxel_spacings:
            voxel_size_input.addItem(str(voxel_spacing.meta.voxel_size))
        layout.addRow("Voxel Size:", voxel_size_input)

        create_button = QPushButton("Create", widget)
        create_button.clicked.connect(
            lambda: self.create_segmentation(
                widget,
                run,
                name_input.text(),
                session_input.value(),
                user_input.text(),
                float(voxel_size_input.currentText()),
            )
        )
        layout.addWidget(create_button)

        self.viewer.window.add_dock_widget(widget, area="right")

    def show_picks_widget(self, run):
        widget = QWidget()
        widget.setWindowTitle("Create New Picks")

        layout = QFormLayout(widget)
        object_name_input = QComboBox(widget)
        for obj in self.root.config.pickable_objects:
            object_name_input.addItem(obj.name)
        layout.addRow("Object Name:", object_name_input)

        session_input = QSpinBox(widget)
        session_input.setValue(0)
        layout.addRow("Session ID:", session_input)

        user_input = QLineEdit(widget)
        user_input.setText("napariCopick")
        layout.addRow("User ID:", user_input)

        create_button = QPushButton("Create", widget)
        create_button.clicked.connect(
            lambda: self.create_picks(
                widget,
                run,
                object_name_input.currentText(),
                session_input.value(),
                user_input.text(),
            )
        )
        layout.addWidget(create_button)

        self.viewer.window.add_dock_widget(widget, area="right")

    def create_segmentation(
        self, widget, run, name, session_id, user_id, voxel_size
    ):
        seg = run.new_segmentation(
            voxel_size=voxel_size,
            name=name,
            session_id=str(session_id),
            is_multilabel=True,
            user_id=user_id,
        )

        tomo = zarr.open(run.voxel_spacings[0].tomograms[0].zarr(), "r")[
            "0"
        ]

        shape = tomo.shape
        dtype = np.int32

        # Create an empty Zarr array for the segmentation
        zarr_file = zarr.open(seg.zarr(), mode="w")
        zarr_file.create_dataset(
            "data",
            shape=shape,
            dtype=dtype,
            chunks=(128, 128, 128),
            fill_value=0,
        )

        self.populate_tree()
        widget.close()

    def create_picks(self, widget, run, object_name, session_id, user_id):
        run.new_picks(
            object_name=object_name,
            session_id=str(session_id),
            user_id=user_id,
        )
        self.populate_tree()
        widget.close()

    def refresh_tree(self):
        self.root = copick.from_file(self.copick_config_path)
        self.populate_tree()

    def populate_solution_dropdown(self):
        self.solution_dropdown.clear()
        try:
            print(f"Fetching solutions from http://{self.hostname}:{self.port}/index")
            response = requests.get(f"http://{self.hostname}:{self.port}/index")
            print(f"Response status: {response.status_code}")
            if response.status_code == 200:
                index = response.json().get('index', {})
                print(f"Number of solutions received: {len(index)}")
                for solution_id, solution_info in index.items():
                    item_text = f"{solution_info['catalog']}:{solution_info['group']}:{solution_info['name']}:{solution_info['version']}"
                    print(f"Adding solution: {item_text}")
                    self.solution_dropdown.addItem(item_text)
            else:
                print(f"Error response content: {response.text}")
        except Exception as e:
            print(f"Error fetching solutions: {e}")

        print(f"Number of items in dropdown after population: {self.solution_dropdown.count()}")
        
    def fetch_models(self):
        """Fetch available models from the server."""
        try:
            response = requests.get(f"http://{self.hostname}:{self.port}/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', {})
                return models
            else:
                print(f"Failed to fetch models. Status code: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error fetching models: {e}")
            return {}

    def populate_model_dropdown(self, combobox):
        models = self.fetch_models()
        combobox.clear()
        for model_path, model_info in models.items():
            combobox.addItem(f"{model_path} ({model_info['catalog']}:{model_info['group']}:{model_info['name']}:{model_info['version']})")


    def fetch_user_and_session_ids(self):
        """Fetch user_id and session_id from the server."""
        try:
            response = requests.get(f"http://{self.hostname}:{self.port}/user_session_ids")
            if response.status_code == 200:
                data = response.json()
                user_ids = data.get('user_ids', [])
                session_ids = data.get('session_ids', [])
                return user_ids, session_ids
            else:
                print(f"Failed to fetch user and session IDs. Status code: {response.status_code}")
                return [], []
        except Exception as e:
            print(f"Error fetching user and session IDs: {e}")
            return [], []

    def update_solution_args(self):
        # Clear existing widgets
        for i in reversed(range(self.scroll_layout.count())): 
            widget = self.scroll_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()  # Ensures the widget is fully cleaned up
                self.scroll_layout.removeWidget(widget)

        selected_run = self.run_dropdown.currentText()
        selected_solution = self.solution_dropdown.currentText()

        default_value = ""

        if not selected_solution:
            return

        catalog, group, name, version = selected_solution.split(":")

        # Define the conditions for freeform parameters
        freeform_conditions = {
            "cellcanvas": {
                "generate-pixel-embedding": ["embedding_name"],
                "generate-tomogram": ["run_name"],
                "generate-skimage-features": ["feature_type"],
                "generate-torch-basic-features": ["feature_type"],
            }
        }

        # Fetch user_id and session_id from the server
        user_ids, session_ids = self.fetch_user_and_session_ids()

        try:
            response = requests.get(f"http://{self.hostname}:{self.port}/info/{catalog}/{group}/{name}/{version}")
            if response.status_code == 200:
                solution_info = response.json().get('info', {})
                args = solution_info.get('args', [])

                for arg in args:
                    arg_name = arg.get('name')

                    # Suppress copick_config_path argument
                    if arg_name == 'copick_config_path':
                        continue

                    label = QLabel(arg_name)

                    # Check if the argument should be freeform
                    if catalog in freeform_conditions and name in freeform_conditions[catalog] and arg_name in freeform_conditions[catalog][name]:
                        field = QLineEdit(str(default_value))
                    else:
                        if arg_name == 'run_name':
                            field = QComboBox()
                            for run in self.root.runs:
                                field.addItem(run.meta.name)
                            field.setCurrentText(selected_run)
                        elif arg_name == 'voxel_spacing':
                            field = QComboBox()
                            run = self.root.get_run(selected_run)
                            for voxel_spacing in run.voxel_spacings:
                                field.addItem(str(voxel_spacing.meta.voxel_size))
                        elif arg_name == 'tomo_type':
                            field = QComboBox()
                            run = self.root.get_run(selected_run)
                            for tomogram in run.voxel_spacings[0].tomograms:
                                field.addItem(tomogram.meta.tomo_type)
                        elif arg_name in ['embedding_name', 'feature_type', 'feature_names']:
                            field = QComboBox()
                            run = self.root.get_run(selected_run)
                            for voxel_spacing in run.voxel_spacings:
                                for tomogram in voxel_spacing.tomograms:
                                    for feature in tomogram.features:
                                        field.addItem(feature.feature_type)
                        elif arg_name in ['user_id']:
                            field = QComboBox()
                            for user_id in user_ids:
                                field.addItem(user_id)
                        elif arg_name in ['session_id']:
                            field = QComboBox()
                            for session_id in session_ids:
                                field.addItem(session_id)
                        elif arg_name == 'model_path':
                            field = QComboBox()
                            self.populate_model_dropdown(field)
                        elif arg_name == 'segmentation_name':
                            field = QComboBox()
                            run = self.root.get_run(selected_run)
                            for voxel_spacing in run.voxel_spacings:
                                for segmentation in voxel_spacing.run.get_segmentations(voxel_spacing.meta.voxel_size):
                                    field.addItem(segmentation.meta.name)
                        else:
                            field = QLineEdit(str(default_value))

                        self.scroll_layout.addRow(label, field)

                    # Ensure the widgets are updated to avoid event filter issues
                    self.scroll_content.update()
        except Exception as e:
            print(f"Error updating solution args: {e}")

    def run_solution(self):
        selected_solution = self.solution_dropdown.currentText()
        if not selected_solution:
            return

        catalog, group, name, version = selected_solution.split(":")
        solution_args = {}
        
        for i in range(self.scroll_layout.rowCount()):
            label_item = self.scroll_layout.itemAt(i * 2)
            field_item = self.scroll_layout.itemAt(i * 2 + 1)
            if label_item and field_item:
                label = label_item.widget()
                field = field_item.widget()
                if isinstance(field, QComboBox):
                    solution_args[label.text()] = field.currentText()
                else:
                    solution_args[label.text()] = field.text()
        
        try:
            response = requests.post(f"http://{self.hostname}:{self.port}/run/{catalog}/{group}/{name}/{version}", 
                                     json={"args": solution_args})
            if response.status_code == 200:
                result = response.json()
                print(f"Execution result: {result}")
            else:
                print(f"Failed to execute solution. Status code: {response.status_code}")
                print(f"Response content: {response.text}")
        except Exception as e:
            print(f"Error occurred: {e}")

def main():
    viewer = napari.Viewer()
    widget = CellCanvasWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget, area='right')
    napari.run()

if __name__ == "__main__":
    main()
