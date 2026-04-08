"""Build TIQS documentation with pdoc."""

from pathlib import Path
import pkgutil
import sys

sys.path.insert(0, "src")

import pdoc
import pdoc.render

import tiqs

pdoc.render.configure(
    docformat="numpy",
    math=True,
    mermaid=True,
    template_directory=Path("docs/templates"),
    footer_text="TIQS - Trapped Ion Quantum Simulator",
)

# Collect only subpackage names (not leaf modules, which pdoc
# auto-discovers from the parent package).
subpackages = [
    modname
    for _importer, modname, ispkg in pkgutil.walk_packages(
        tiqs.__path__, prefix="tiqs."
    )
    if ispkg
]

# Pass each subpackage plus standalone modules so pdoc generates
# a separate page per subpackage with its theory docstring visible.
modules = [
    *subpackages,
    "tiqs.constants",
    "tiqs.trap",
    "tiqs.transport",
]

pdoc.pdoc(*modules, output_directory=Path("docs/api"))
