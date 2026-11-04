# napari-cellcanvas

[![License MIT](https://img.shields.io/pypi/l/napari-cellcanvas.svg?color=green)](https://github.com/cellcanvas/napari-cellcanvas/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-cellcanvas.svg?color=green)](https://pypi.org/project/napari-cellcanvas)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-cellcanvas.svg?color=green)](https://python.org)
[![tests](https://github.com/cellcanvas/napari-cellcanvas/workflows/tests/badge.svg)](https://github.com/cellcanvas/napari-cellcanvas/actions)
[![codecov](https://codecov.io/gh/kephale/napari-cellcanvas/branch/main/graph/badge.svg)](https://codecov.io/gh/kephale/napari-cellcanvas)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cellcanvas)](https://napari-hub.org/plugins/napari-cellcanvas)

A napari plugin for interfacing with CellCanvas

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-cellcanvas` via [pip]:

    pip install 'napari-cellcanvas[qt]'



To install latest development version :

    pip install 'napari-cellcanvas[qt] @ git+https://github.com/cellcanvas/napari-cellcanvas.git'


Requires Python 3.11+. Open CellCanvas from the napari plugin menu.
For local Zarr 3 datasets, install `[qt,local]` and choose **Open Local Dataset**.
Select image, feature, painting and prediction stores, then **Train** and **Predict and Save**.
Copick browsing uses `[qt,copick]` in a separate Zarr 2 environment.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-cellcanvas" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/cellcanvas/napari-cellcanvas/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
