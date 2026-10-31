import json
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread, get_ident
from types import SimpleNamespace

import pytest
import requests
from qtpy.QtCore import QThread, QTimer
from qtpy.QtWidgets import QFormLayout

from napari_cellcanvas._requests import RequestRunner
from napari_cellcanvas._server import ServerError, SolutionClient
from napari_cellcanvas.widget import CellCanvasWidget


SOLUTION = "catalog:group:first:1"


def wait_for(app, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition():
        app.processEvents()
        if time.monotonic() > deadline:
            pytest.fail("Timed out waiting for a GUI result")
        time.sleep(0.001)
    app.processEvents()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    state = SimpleNamespace(routes={
        "/index": {"index": {"first": {"catalog": "catalog", "group": "group", "name": "first", "version": "1"}}},
        "/info/catalog/group/first/1": {"info": {"args": []}},
        "/models": {"models": ["model-a"]},
        "/run/catalog/group/first/1": {"result": "ok"},
    }, calls=[])

    class Handler(BaseHTTPRequestHandler):
        def answer(self):
            payload = None
            if self.command == "POST":
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append((self.command, self.path, payload))
            response = state.routes.get(self.path, (404, {"error": "missing"}))
            if callable(response):
                response = response()
            status, body = response if isinstance(response, tuple) else (200, response)
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = answer
        do_POST = answer

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.port = http.server_port
    state.url = f"http://127.0.0.1:{state.port}"
    Thread(target=lambda: http.serve_forever(poll_interval=0.01), daemon=True).start()
    yield state
    http.shutdown()
    http.server_close()


def hold(response):
    started, release = Event(), Event()

    def reply():
        started.set()
        release.wait(2)
        return response

    return reply, started, release


@pytest.fixture
def widget(app, server):
    result = CellCanvasWidget(hostname="127.0.0.1", port=server.port, request_timeout=(0.2, 2))
    yield result
    result.close()
    result.deleteLater()
    app.processEvents()


def test_transport_has_no_viewer_imports():
    subprocess.run([sys.executable, "-c", "from napari_cellcanvas._server import SolutionClient; "
                    "import sys; assert 'napari' not in sys.modules; assert 'qtpy' not in sys.modules"], check=True)


def test_existing_protocol_and_execution_result_are_preserved(server):
    server.routes["/info/catalog/group/first/1"] = {"info": {"args": [{"name": "value"}]}}
    server.routes["/run/catalog/group/first/1"] = ["opaque", "result"]
    client = SolutionClient(server.url)
    assert client.solutions() == [SOLUTION]
    assert client.arguments(SOLUTION) == [{"name": "value"}]
    assert client.models() == ["model-a"]
    assert client.run(SOLUTION, {"value": "123"}) == ["opaque", "result"]
    assert server.calls == [
        ("GET", "/index", None), ("GET", "/info/catalog/group/first/1", None),
        ("GET", "/models", None), ("POST", "/run/catalog/group/first/1", {"args": {"value": "123"}}),
    ]


def test_delayed_server_keeps_gui_timer_running(app, widget, server):
    reply, started, release = hold(server.routes["/index"])
    server.routes["/index"] = reply
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(5)
    before = time.monotonic()
    widget.populate_solution_dropdown()
    assert time.monotonic() - before < 0.25
    wait_for(app, lambda: started.is_set() and len(ticks) >= 5)
    assert not release.is_set()
    assert not widget.refresh_solutions_button.isEnabled()
    release.set()
    wait_for(app, widget.run_button.isEnabled)
    assert widget.refresh_solutions_button.isEnabled()
    timer.stop()


@pytest.mark.parametrize("response", [b"{broken", {"index": []}, {"index": {"x": {}}}, (503, {})])
def test_server_errors_are_visible_and_retry_recovers(app, widget, server, response):
    good = server.routes["/index"]
    server.routes["/index"] = response
    widget.populate_solution_dropdown()
    wait_for(app, widget.refresh_solutions_button.isEnabled)
    assert not widget.run_button.isEnabled()
    assert widget.solution_dropdown.isEnabled()
    assert widget.status_label.text() != "Loading solutions..."
    assert widget.status_label.text() != "Solution ready."
    server.routes["/index"] = good
    widget.populate_solution_dropdown()
    wait_for(app, widget.run_button.isEnabled)


def test_unreachable_server_recovers(app, widget, server):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        widget.port = sock.getsockname()[1]
    widget.populate_solution_dropdown()
    wait_for(app, widget.refresh_solutions_button.isEnabled)
    # Windows may reach the connect timeout before reporting a refused port.
    assert any(word in widget.status_label.text().lower() for word in ("connect", "timed out"))
    assert not widget.run_button.isEnabled()
    widget.port = server.port
    widget.populate_solution_dropdown()
    wait_for(app, widget.run_button.isEnabled)


def test_refused_connection_has_clear_error(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", refuse)
    with pytest.raises(ServerError, match="connect"):
        SolutionClient("http://127.0.0.1:1").solutions()


def test_read_timeout_recovers_controls(app, widget, server):
    reply, started, release = hold(server.routes["/index"])
    server.routes["/index"] = reply
    widget._request_timeout = (0.1, 0.1)
    widget.populate_solution_dropdown()
    wait_for(app, widget.refresh_solutions_button.isEnabled)
    assert started.is_set()
    assert "timed out" in widget.status_label.text().lower()
    assert not widget.run_button.isEnabled()
    release.set()


def test_stale_arguments_do_not_replace_new_selection(app, widget, server):
    reply, started, release = hold({"info": {"args": [{"name": "old"}]}})
    server.routes["/info/catalog/group/first/1"] = reply
    server.routes["/info/catalog/group/second/1"] = {"info": {"args": [{"name": "new"}]}}
    completions = []
    widget._requests.completed.connect(lambda *args: completions.append(args))
    widget.solution_dropdown.blockSignals(True)
    widget.solution_dropdown.addItems([SOLUTION, "catalog:group:second:1"])
    widget.solution_dropdown.blockSignals(False)
    widget.update_solution_args()
    wait_for(app, started.is_set)
    widget.solution_dropdown.setCurrentIndex(1)
    wait_for(app, widget.run_button.isEnabled)
    release.set()
    wait_for(app, lambda: len([r for r in completions if r[0] == "arguments"]) == 2)
    assert widget.scroll_layout.rowCount() == 1
    assert widget.scroll_layout.itemAt(0, QFormLayout.LabelRole).widget().text() == "new"
    assert widget.status_label.text() == "Solution ready."


def test_stale_models_do_not_touch_deleted_fields(app, widget, server):
    server.routes["/info/catalog/group/first/1"] = {"info": {"args": [{"name": "model_path"}]}}
    server.routes["/info/catalog/group/second/1"] = {"info": {"args": []}}
    reply, started, release = hold({"models": ["old-model"]})
    server.routes["/models"] = reply
    completions = []
    widget._requests.completed.connect(lambda *args: completions.append(args))
    widget.solution_dropdown.blockSignals(True)
    widget.solution_dropdown.addItems([SOLUTION, "catalog:group:second:1"])
    widget.solution_dropdown.blockSignals(False)
    widget.update_solution_args()
    wait_for(app, started.is_set)
    widget.solution_dropdown.setCurrentIndex(1)
    wait_for(app, widget.run_button.isEnabled)
    release.set()
    wait_for(app, lambda: any(r[0] == "models" for r in completions))
    assert widget.scroll_layout.rowCount() == 0
    assert widget.run_button.isEnabled()
    assert widget.status_label.text() == "Solution ready."


def test_run_uses_current_fields_and_updates_on_gui_thread(app, widget, server, monkeypatch):
    server.routes["/info/catalog/group/first/1"] = {"info": {"args": [{"name": "value"}, {"name": "model_path"}]}}
    original = widget._render_solution_args
    threads = []

    def render(*args):
        threads.append(QThread.currentThread())
        original(*args)

    monkeypatch.setattr(widget, "_render_solution_args", render)
    widget.populate_solution_dropdown()
    wait_for(app, widget.run_button.isEnabled)
    widget.scroll_layout.itemAt(0, QFormLayout.FieldRole).widget().setText("123")
    widget.run_button.click()
    assert not widget.run_button.isEnabled()
    wait_for(app, lambda: widget.last_solution_result is not None)
    assert widget.run_button.isEnabled()
    assert widget.last_solution_result == {"result": "ok"}
    assert threads == [app.thread()]
    assert server.calls[-1] == ("POST", "/run/catalog/group/first/1", {"args": {"value": "123", "model_path": "model-a"}})


def test_closing_does_not_wait_for_network_or_apply_late_results(app, widget, server, monkeypatch):
    reply, started, release = hold(server.routes["/index"])
    server.routes["/index"] = reply
    completions = []
    done = Event()
    client = widget._client()
    original = client.solutions

    def work():
        try:
            return original()
        finally:
            done.set()

    client.solutions = work
    monkeypatch.setattr(widget, "_client", lambda: client)
    widget._requests.completed.connect(lambda *args: completions.append(args))
    widget.populate_solution_dropdown()
    wait_for(app, started.is_set)
    before = time.monotonic()
    widget.close()
    assert time.monotonic() - before < 0.1
    assert widget._requests._closed.is_set()
    release.set()
    wait_for(app, done.is_set)
    assert completions == []


def test_failed_run_can_be_retried(app, widget, server):
    widget.populate_solution_dropdown()
    wait_for(app, widget.run_button.isEnabled)
    server.routes["/run/catalog/group/first/1"] = (500, {})
    widget.run_button.click()
    wait_for(app, lambda: not widget._running_solution)
    assert "500" in widget.status_label.text()
    assert widget.run_button.isEnabled()
    server.routes["/run/catalog/group/first/1"] = {"result": "retried"}
    widget.run_button.click()
    wait_for(app, lambda: not widget._running_solution)
    assert widget.last_solution_result == {"result": "retried"}


def test_run_result_does_not_overwrite_new_selection(app, widget, server):
    widget.populate_solution_dropdown()
    wait_for(app, widget.run_button.isEnabled)
    reply, started, release = hold({"result": "old"})
    server.routes["/run/catalog/group/first/1"] = reply
    server.routes["/info/catalog/group/second/1"] = {"info": {"args": []}}
    widget.run_button.click()
    wait_for(app, started.is_set)
    widget.run_solution()  # A second click/API call must not duplicate execution.
    widget.solution_dropdown.addItem("catalog:group:second:1")
    widget.solution_dropdown.setCurrentIndex(1)
    wait_for(app, lambda: widget._args_ready)
    release.set()
    wait_for(app, lambda: not widget._running_solution)
    assert widget.last_solution_result is None
    assert widget.status_label.text() == "Solution ready."
    assert widget.run_button.isEnabled()
    assert len([call for call in server.calls if call[0] == "POST"]) == 1


@pytest.mark.parametrize("endpoint,payload", [
    ("/info/catalog/group/first/1", {"info": {"args": "invalid"}}),
    ("/models", {"models": "invalid"}),
])
def test_argument_and_model_failures_leave_controls_recoverable(app, widget, server, endpoint, payload):
    server.routes["/info/catalog/group/first/1"] = {"info": {"args": [{"name": "model_path"}]}}
    original = server.routes[endpoint]
    server.routes[endpoint] = payload
    widget.populate_solution_dropdown()
    wait_for(app, lambda: "must be" in widget.status_label.text())
    assert widget.refresh_solutions_button.isEnabled()
    assert not widget.run_button.isEnabled()
    server.routes[endpoint] = original
    widget.populate_solution_dropdown()
    wait_for(app, widget.run_button.isEnabled)


def test_worker_capacity_is_bounded_and_callbacks_use_gui_thread(app):
    runner = RequestRunner(max_workers=1)
    release = Event()
    results, failures = [], []
    runner.submit("slow", lambda: release.wait(2), results.append, failures.append)
    runner.submit("busy", lambda: "must not start", results.append, failures.append)
    assert len(failures) == 1 and "busy" in failures[0]
    release.set()
    wait_for(app, lambda: len(results) == 1)
    runner.submit("thread", get_ident,
                  lambda worker: results.append((worker, get_ident())), failures.append)
    wait_for(app, lambda: len(results) == 2)
    assert results[1][0] != results[1][1] == get_ident()
    runner.close()


@pytest.mark.parametrize("payload", [{"models": {}}, {"info": {"args": [{"name": "x"}, {"name": "x"}]}}])
def test_client_rejects_malformed_metadata(server, payload):
    client = SolutionClient(server.url)
    endpoint, operation = ("/models", client.models) if "models" in payload else (
        "/info/catalog/group/first/1", lambda: client.arguments(SOLUTION))
    server.routes[endpoint] = payload
    with pytest.raises(ServerError):
        operation()
