"""Qt dispatch for plain Python requests; never wait for a worker on close."""

from threading import Event, Semaphore, Thread

from qtpy.QtCore import QObject, Qt, Signal, Slot


class RequestRunner(QObject):
    completed = Signal(str, int, object, object)

    def __init__(self, parent=None, max_workers=4):
        super().__init__(parent)
        self._closed = Event()
        closed = self._closed
        self.destroyed.connect(lambda *_: closed.set())
        self._slots = Semaphore(max_workers)
        self._serial = 0
        self._callbacks = {}
        self.completed.connect(self._finish, Qt.QueuedConnection)

    def submit(self, key, work, success, failure, finished=lambda: None):
        if self._closed.is_set():
            return
        self._serial += 1
        token = self._serial
        self._callbacks[key] = (token, success, failure, finished)
        if not self._slots.acquire(blocking=False):
            self._finish(key, token, None, "Server is busy; retry after the current requests finish.")
            return
        closed, slots = self._closed, self._slots

        def run():
            try:
                result, error = work(), None
            except Exception as exception:
                result, error = None, str(exception)
            finally:
                slots.release()
            if not closed.is_set():
                try:
                    self.completed.emit(key, token, result, error)
                except RuntimeError:
                    # The parent may have been deleted between the check and emit.
                    pass

        Thread(target=run, daemon=True).start()

    def invalidate(self, *keys):
        for key in keys:
            self._callbacks.pop(key, None)

    @Slot(str, int, object, object)
    def _finish(self, key, token, result, error):
        callbacks = self._callbacks.get(key)
        if self._closed.is_set() or callbacks is None or callbacks[0] != token:
            return
        _, success, failure, finished = self._callbacks.pop(key)
        try:
            if error is None:
                success(result)
            else:
                failure(error)
        except Exception as exception:
            failure(str(exception))
        finally:
            finished()

    def close(self):
        """Discard callbacks without waiting for in-flight network I/O."""
        self._closed.set()
        self._callbacks.clear()
