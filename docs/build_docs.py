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
    template_directory=Path("docs/templates"),
    footer_text="TIQS - Trapped Ion Quantum Simulator",
)

# pdoc collapses all submodules into the top-level page unless
# subpackages are passed explicitly. Discover them automatically
# so each subpackage gets its own page with theory docstrings.
modules = ["tiqs"]
for _importer, modname, ispkg in pkgutil.walk_packages(
    tiqs.__path__, prefix="tiqs."
):
    if ispkg:
        modules.append(modname)

pdoc.pdoc(*modules, output_directory=Path("docs/api"))
