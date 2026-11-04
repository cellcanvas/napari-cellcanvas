__version__ = "0.0.1"

__all__ = (
    "CellCanvasWidget",
)


def __getattr__(name):
    if name == "CellCanvasWidget":
        from .widget import CellCanvasWidget
        return CellCanvasWidget
    raise AttributeError(name)
