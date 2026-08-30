"""Runtime cross-package import gate (PS-140).

Every module listed here is imported by this package's source but OWNED
by a peer standalone. A rename or move on the other side of that boundary
is invisible to this package's unit tests and surfaces as
ModuleNotFoundError in a user's process — which is how the
`scitex_io._load_cache` rename went undetected for weeks.

The import LIST between the sentinels is generated; regenerate it with
`scitex-dev ecosystem install-cross-package-gate scitex-notebook --force`
(everything below the closing sentinel is preserved byte-for-byte, so
hand-written cases belong there).

Three outcomes, and the middle one is the only reason this file exists:

- Peer distribution installed AND the full dotted path imports → PASSES.
- Peer distribution installed BUT the submodule import fails (e.g. an
  internal rename like `scitex_io._load_cache` →
  `scitex_io._loading._load_cache`) → FAILS LOUDLY.
- Peer distribution NOT installed (lean install, optional extra, a
  marker-gated dependency) → SKIPPED on the ROOT package.

That middle outcome only holds because the skip is scoped to the ROOT.
`pytest.importorskip(module_name)` on the FULL dotted path skips on any
ImportError, and a renamed submodule raises ModuleNotFoundError — an
ImportError subclass. Under that spelling the rename SKIPS, the hard
import is never reached, and the gate reports green: a gate that cannot
fail, which is the same thing as a deleted one.
"""

import importlib

import pytest

# ===== AUTO-GENERATED: cross-package imports =====
CROSS_PACKAGE_IMPORTS = [
    "scitex_clew",
    "scitex_clew._tracker",
    "scitex_context",
    "scitex_dev._cli._completion",
]
# ===== END AUTO-GENERATED =====


@pytest.mark.parametrize("module_name", CROSS_PACKAGE_IMPORTS)
def test_cross_package_import(module_name):
    """Importing scitex-notebook's declared cross-package dependency must succeed."""
    # Arrange — skip on the ROOT, and only on the ROOT. Banning the skip
    # outright would convert a legitimate absence into a hard failure — a
    # gate that cannot PASS, in place of one that cannot FAIL. A lean
    # install where a peer distribution is genuinely absent must SKIP here.
    #
    # Two statements ON PURPOSE. The intermediate binding is what makes the
    # root/full-path distinction visible to a reader, which is the entire
    # point of the shape; inlining it to satisfy a checker would make this
    # file harder to read.
    root = module_name.split(".")[0]
    pytest.importorskip(root)

    # Act — a real import of the FULL dotted path. Not
    # importlib.util.find_spec, which only proves a module is FINDABLE
    # while the failures this gate exists to catch (a renamed symbol
    # re-exported through a package __init__) happen at EXECUTION. And not
    # importorskip(module_name), which skips on the full path and so
    # reports the rename as an absence.
    module = importlib.import_module(module_name)

    # Assert
    assert module is not None
