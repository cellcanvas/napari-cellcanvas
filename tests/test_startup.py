import json
from importlib.resources import files
from unittest.mock import Mock

import pytest
import requests
from npe2 import PluginManager, PluginManifest

from napari_cellcanvas.widget import CellCanvasWidget
import napari_cellcanvas.widget as widget_module


@pytest.fixture(autouse=True)
def no_server(monkeypatch):
    get = Mock(side_effect=AssertionError("unexpected server request"))
    post = Mock(side_effect=AssertionError("unexpected server request"))
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    yield
    get.assert_not_called()
    post.assert_not_called()


@pytest.fixture
def widget(app):
    result = CellCanvasWidget()
    yield result
    result.close()
    result.deleteLater()
    app.processEvents()


@pytest.fixture
def project(tmp_path):
    copick = pytest.importorskip("copick")
    config = tmp_path / "project.json"
    config.write_text(json.dumps({
        "config_type": "filesystem", "name": "Startup test", "description": "test",
        "version": "1.0.0", "pickable_objects": [],
        "overlay_root": str(tmp_path / "overlay"), "overlay_fs_args": {"auto_mkdir": True},
    }))
    root = copick.from_file(str(config))
    root.new_run("run-one")
    return config


def test_manifest_resolves_and_opens_widget(app):
    manifest = PluginManifest.from_file(str(files("napari_cellcanvas") / "napari.yaml"))
    manager = PluginManager()
    try:
        manager.get_manifest(manifest.name)
    except KeyError:
        manager.register(manifest)
    manager.activate(manifest.name)
    assert manifest.contributions.widgets[0].command == "napari-cellcanvas.make_widget"
    factory = manager.commands.get("napari-cellcanvas.make_widget")
    result = factory()
    assert isinstance(result, CellCanvasWidget)
    assert result.root is None
    result.close()


def test_empty_startup_does_not_import_copick(app, monkeypatch):
    importer = Mock(side_effect=AssertionError("unexpected Copick import"))
    monkeypatch.setattr(widget_module, "import_module", importer)
    result = CellCanvasWidget()
    assert result.root is None
    assert result.copick_config_path is None
    assert result.tree_view.topLevelItemCount() == 0
    assert result.run_dropdown.count() == result.solution_dropdown.count() == 0
    assert not result.run_button.isEnabled()
    assert not result.refresh_solutions_button.isEnabled()
    assert not result.local_controls.train_button.isEnabled()
    assert not result.local_controls.predict_button.isEnabled()
    importer.assert_not_called()
    result.close()


def test_refresh_without_project_is_safe(widget):
    widget.refresh_button.click()
    assert widget.root is None
    assert widget.tree_view.topLevelItemCount() == 0


def test_cancel_project_dialog_preserves_state(widget, monkeypatch):
    monkeypatch.setattr(widget_module.QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    widget.open_project_button.click()
    assert widget.root is None


def test_missing_copick_has_visible_feedback(widget, monkeypatch):
    monkeypatch.setattr(widget_module, "import_module", Mock(side_effect=ModuleNotFoundError("copick")))
    assert widget.open_project("project.json") is False
    assert "napari-cellcanvas[copick]" in widget.status_label.text()
    assert widget.root is None


def test_valid_project_selection(widget, project, monkeypatch):
    monkeypatch.setattr(widget_module.QFileDialog, "getOpenFileName", lambda *args: (str(project), ""))
    widget.open_project_button.click()
    assert widget.root.config.name == "Startup test"
    assert widget.copick_config_path == str(project)
    assert widget.tree_view.topLevelItemCount() == 1
    assert widget.tree_view.topLevelItem(0).text(0) == "run-one"
    assert widget.run_dropdown.currentText() == "run-one"
    assert widget.refresh_solutions_button.isEnabled()
    assert not widget.run_button.isEnabled()


def test_invalid_project_preserves_loaded_project(widget, project, tmp_path):
    assert widget.open_project(project)
    previous = widget.root
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json")
    assert widget.open_project(invalid) is False
    assert "Could not open project" in widget.status_label.text()
    assert widget.root is previous
    assert widget.copick_config_path == str(project)
    assert widget.run_dropdown.currentText() == "run-one"


def test_constructor_accepts_explicit_project(app, project):
    result = CellCanvasWidget(copick_config_path=project)
    assert result.root.config.name == "Startup test"
    assert result.run_dropdown.currentText() == "run-one"
    result.close()


def test_refresh_reloads_project_without_contacting_server(widget, project):
    assert widget.open_project(project)
    previous = widget.root
    widget.refresh_button.click()
    assert widget.root is not previous
    assert widget.run_dropdown.currentText() == "run-one"
