"""Plain Python client for the existing Album-style solution endpoints.

No viewer or Qt objects cross this boundary. This protocol is separate from
the newer CellCanvas painting API.
"""

from urllib.parse import quote
from math import isfinite

import requests


class ServerError(RuntimeError):
    pass


class SolutionClient:
    def __init__(self, base_url, timeout=(1.0, 10.0)):
        if len(timeout) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or
            not isfinite(value) or value <= 0 for value in timeout
        ):
            raise ValueError("Connect and read timeouts must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = tuple(timeout)

    def _request(self, method, path, payload=None):
        try:
            if method == "GET":
                response = requests.get(self.base_url + path, timeout=self.timeout)
            else:
                response = requests.post(self.base_url + path, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as error:
            raise ServerError("Server timed out. Retry or check the server.") from error
        except requests.ConnectionError as error:
            raise ServerError("Could not connect to the server. Check its address and port.") from error
        except requests.HTTPError as error:
            raise ServerError(f"Server returned HTTP {error.response.status_code}.") from error
        except requests.RequestException as error:
            raise ServerError("Server request failed. Check the server settings.") from error
        try:
            return response.json()
        except ValueError as error:
            raise ServerError("Server returned malformed JSON.") from error

    @staticmethod
    def _solution_path(solution):
        parts = solution.split(":")
        if len(parts) != 4 or not all(parts):
            raise ServerError("A solution needs catalog, group, name and version")
        return "/".join(quote(part, safe="") for part in parts)

    def solutions(self):
        response = self._request("GET", "/index")
        index = response.get("index") if isinstance(response, dict) else None
        if not isinstance(index, dict):
            raise ServerError("Server index must be an object")
        result = []
        for entry in index.values():
            if not isinstance(entry, dict) or not all(
                isinstance(entry.get(key), str) and entry[key] and ":" not in entry[key]
                for key in ("catalog", "group", "name", "version")
            ):
                raise ServerError("Server index contains an invalid solution")
            result.append(":".join(entry[key] for key in ("catalog", "group", "name", "version")))
        return result

    def models(self):
        response = self._request("GET", "/models")
        models = response.get("models") if isinstance(response, dict) else None
        if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
            raise ServerError("Server models must be a list of names")
        return models

    def arguments(self, solution):
        response = self._request("GET", "/info/" + self._solution_path(solution))
        info = response.get("info") if isinstance(response, dict) else None
        args = info.get("args") if isinstance(info, dict) else None
        if not isinstance(args, list) or not all(
            isinstance(arg, dict) and isinstance(arg.get("name"), str) and arg["name"]
            for arg in args
        ):
            raise ServerError("Server arguments must be a list of named arguments")
        if len({arg["name"] for arg in args}) != len(args):
            raise ServerError("Server argument names must be unique")
        return args

    def run(self, solution, arguments):
        return self._request("POST", "/run/" + self._solution_path(solution), {"args": arguments})
