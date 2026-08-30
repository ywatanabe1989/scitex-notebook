#!/usr/bin/env python3
# Timestamp: "2026-05-02 (ywatanabe)"
# File: src/scitex_notebook/_magic.py
"""IPython extension that detects notebook reproducibility anti-patterns.

Adoption is one line at the top of any notebook::

    %load_ext scitex_notebook

After that, each executed cell is analysed at runtime and recorded in the
same Clew store used by ``@scitex.session`` and ``stx.io``. The
extension targets three of the canonical reproducibility risks for
notebooks identified in the literature:

1. **Hidden-state leak** (Pimentel 2019, Samuel 2024) — a cell reads a name
   that was *not* defined by any previously executed cell in this run, so
   it silently depends on prior kernel state. The extension AST-scans each
   cell, tracks ``(name → defining cell)``, and emits a runtime warning
   the first time a leaky name is loaded.
2. **Out-of-order execution** (Pimentel 2019, Wang 2020) — IPython's
   ``execution_count`` is recorded per cell so a post-run check can verify
   monotonic execution against the on-disk cell order.
3. **Untracked I/O** — every ``stx.io.save/load`` call inside the cell
   attaches to that cell's :class:`SessionTracker` via :func:`set_tracker`,
   so file hashes enter the per-cell record automatically.

The extension also rewrites the ``parent_session`` link to encode the real
data-dependency edge: ``parent`` becomes the most recent earlier cell whose
*defined* names overlap with this cell's *loaded* names. If the cell has no
intra-run dependency, ``parent`` is ``None`` (the cell is independent),
which is itself useful information.

Caveats
-------
* Hidden-state detection is name-level, not value-level — we do not
  snapshot the kernel namespace. A cell that mutates ``df`` in place is
  visible only through the cell that *defined* ``df``.
* AST analysis is a best-effort static approximation: cells using
  ``exec``/``eval``, dynamic ``globals()`` access, or IPython line-magics
  may produce false negatives.
* The extension never edits user code. All findings are advisory warnings
  in stderr plus structured metadata in the Clew DB.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# scitex_clew is a hard dependency: this magic is meaningless without it.
try:
    from scitex_clew._tracker import SessionTracker, set_tracker
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "scitex_notebook._magic requires scitex-clew. "
        "Install with `pip install scitex-clew`."
    ) from e


# Python builtins resolve "for free" in any cell — never flag them as
# leaky. Includes things like ``print``, ``len``, ``True``, etc.
_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))

# IPython injects a few names into ``user_ns`` itself; treat them as
# always-available so we don't flood with false positives.
_IPYTHON_NAMES: frozenset[str] = frozenset(
    {
        "In",
        "Out",
        "_",
        "__",
        "___",
        "_i",
        "_ii",
        "_iii",
        "exit",
        "quit",
        "get_ipython",
    }
)

_AMBIENT_NAMES: frozenset[str] = _BUILTIN_NAMES | _IPYTHON_NAMES


def _short_id(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _ast_loads_and_stores(src: str) -> tuple[set[str], set[str]]:
    """Return ``(loaded_names, stored_names)`` from a single cell.

    Conservative best-effort: returns empty sets on syntax error (e.g. the
    cell is an IPython line-magic like ``%timeit``). ``stored_names``
    includes names introduced by assignment, function/class definitions,
    and imports. ``loaded_names`` contains every ``ast.Name`` with a
    ``Load`` context, with builtins/IPython ambient names filtered out.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), set()

    loads: set[str] = set()
    stores: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loads.add(node.id)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                stores.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores.add(node.name)
        elif isinstance(node, ast.Import):
            for n in node.names:
                stores.add((n.asname or n.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                stores.add(n.asname or n.name)
        elif isinstance(node, ast.Global):
            stores.update(node.names)

    loads -= _AMBIENT_NAMES
    return loads, stores


def _detect_notebook_path(shell) -> Optional[Path]:
    """Best-effort detection of the running notebook's path.

    Order of attempts:

    1. ``scitex_context.get_notebook_path()`` if importable.
    2. ``ipykernel`` + Jupyter server REST query (existing helper).
    3. ``IPYTHONDIR`` / fallback to a synthetic path under ``$PWD``.

    Returns ``None`` if no path can be resolved (we still track, just
    without a stable notebook handle).
    """
    try:
        from scitex_context import get_notebook_path

        nb = get_notebook_path()
        if nb:
            return Path(nb)
    except Exception:
        pass
    # Last-resort fallback: synthesize a stable name from the shell session.
    if shell is not None:
        sess = getattr(shell, "history_manager", None)
        if sess is not None:
            sess_id = getattr(sess, "session_number", 0)
            return Path(os.getcwd()) / f"untitled-session-{sess_id}.ipynb"
    return None


class ScitexNotebookMagics:
    """Lightweight non-Magics IPython hook installer.

    We don't subclass ``IPython.core.magic.Magics`` because we don't expose
    user-callable ``%line`` or ``%%cell`` magics — the entire feature is
    automatic per-cell instrumentation triggered by ``%load_ext``.
    """

    def __init__(self, shell):
        self.shell = shell
        self.notebook_path = _detect_notebook_path(shell)
        self._exec_index = 0
        self._prev_session: Optional[str] = None
        # Per-cell scratch state populated in pre and consumed in post.
        self._tracker: Optional[SessionTracker] = None
        self._cell_t0: Optional[float] = None
        self._cell_src_hash: Optional[str] = None
        self._cell_loads: set[str] = set()
        self._cell_stores: set[str] = set()
        self._cell_warnings: list[dict[str, Any]] = []
        self._cell_dep_parents: list[str] = []

        # Per-run state for hidden-state-leak / dependency-DAG detection.
        # Maps each defined name to the (cell_index, session_id) that first
        # introduced it. Most-recent definition is recorded so re-defining
        # in a later cell shifts the dependency edge.
        self._name_defs: dict[str, tuple[int, str]] = {}

        # Aggregate warnings for post-run summary (visible to the cohort
        # runner via ``magic.warnings_summary()``).
        self.warnings: list[dict[str, Any]] = []

        # Stable per-notebook session-id prefix
        nb_stem = self.notebook_path.stem if self.notebook_path else "untitled-notebook"
        run_tag = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self._sid_prefix = f"nb-{nb_stem}-{run_tag}-{_short_id(nb_stem + run_tag, 6)}"

        shell.events.register("pre_run_cell", self._pre_run_cell)
        shell.events.register("post_run_cell", self._post_run_cell)

    # ------------------------------------------------------------------
    # Public summary API (called by the cohort runner)
    # ------------------------------------------------------------------

    def warnings_summary(self) -> dict[str, Any]:
        """Aggregate per-cell warnings into a runner-friendly dict."""
        kinds: dict[str, int] = {}
        for w in self.warnings:
            kinds[w["kind"]] = kinds.get(w["kind"], 0) + 1
        return {
            "n_cells_executed": self._exec_index,
            "n_warnings": len(self.warnings),
            "by_kind": kinds,
            "warnings": self.warnings,
        }

    # ------------------------------------------------------------------
    # IPython hooks
    # ------------------------------------------------------------------

    def _pre_run_cell(self, info) -> None:
        """Open a tracker for this cell and install it as the global one."""
        self._exec_index += 1
        src = getattr(info, "raw_cell", "") or ""
        self._cell_src_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()
        self._cell_t0 = time.time()

        # Static analysis of this cell's name graph.
        loads, stores = _ast_loads_and_stores(src)
        self._cell_loads = loads
        self._cell_stores = stores
        self._cell_warnings = []
        self._cell_dep_parents = []

        # 1. Hidden-state-leak detection (Pimentel 2019, Samuel 2024).
        leaky_names = sorted(loads - stores - set(self._name_defs.keys()))
        for name in leaky_names:
            warn = {
                "kind": "hidden_state_leak",
                "cell_index": self._exec_index,
                "name": name,
                "message": (
                    f"cell {self._exec_index} reads name {name!r} that was "
                    "not defined by any earlier cell in this run; the cell "
                    "depends on prior kernel state and may not reproduce "
                    "from a fresh kernel."
                ),
            }
            self._cell_warnings.append(warn)
            self.warnings.append(warn)
            print(
                f"[scitex-notebook] WARN cell {self._exec_index}: "
                f"hidden-state name {name!r} (no defining cell in this run)",
                file=sys.stderr,
            )

        # 2. Data-dependency parent edge — most recent earlier cell that
        #    defined a name this cell loads. Falls back to ``None``
        #    (independent cell) rather than the trivial previous-executed
        #    cell, because the latter encodes nothing reproducibility-wise.
        dep_parents: list[tuple[int, str]] = []
        for name in loads:
            if name in self._name_defs:
                dep_parents.append(self._name_defs[name])
        # Sort by cell_index descending → most recent definer first.
        dep_parents.sort(key=lambda t: -t[0])
        primary_parent_sid: Optional[str] = dep_parents[0][1] if dep_parents else None
        self._cell_dep_parents = [sid for _, sid in dep_parents]

        sid = f"{self._sid_prefix}-cell-{self._exec_index:04d}"

        metadata: dict[str, Any] = {
            "scitex_notebook_magic_version": 2,
            "notebook_path": str(self.notebook_path) if self.notebook_path else None,
            "cell_index": self._exec_index,
            "cell_source_sha256": self._cell_src_hash,
            "cell_source_len": len(src),
            "loads": sorted(loads),
            "stores": sorted(stores),
            "leaky_names": leaky_names,
            "dep_parents": self._cell_dep_parents,
        }

        self._tracker = SessionTracker(
            session_id=sid,
            script_path=str(self.notebook_path) if self.notebook_path else None,
            parent_session=primary_parent_sid,
            metadata=metadata,
        )
        set_tracker(self._tracker)

    def _post_run_cell(self, result) -> None:
        """Finalize the cell's tracker and chain forward."""
        if self._tracker is None:
            return
        try:
            had_error = getattr(result, "error_in_exec", None) is not None
            status = "failed" if had_error else "success"
            exit_code = 1 if had_error else 0
            # Wall time and error flag are not yet persisted: SessionTracker
            # accepts metadata only at construction in this scitex-clew rev.
            # Attaching them as a follow-up requires a small _db.update_run
            # API; left as TODO so this magic stays a one-file change.
            self._tracker.finalize(status=status, exit_code=exit_code)

            # 3. Out-of-order execution check (Wang 2020, Pimentel 2019).
            #    IPython's execution_count is the global cell-run counter.
            #    In a clean linear run starting from a fresh kernel it
            #    increases by exactly 1 per cell and matches our local
            #    self._exec_index. Mismatches mean the user re-ran cells
            #    or skipped some — both reproducibility risks.
            ec = getattr(result, "execution_count", None)
            if ec is not None and ec != self._exec_index:
                warn = {
                    "kind": "out_of_order_execution",
                    "cell_index": self._exec_index,
                    "execution_count": ec,
                    "message": (
                        f"cell {self._exec_index} ran with execution_count={ec}; "
                        "kernel was not fresh — earlier cells were re-run or "
                        "skipped, breaking linear-from-top reproducibility."
                    ),
                }
                self._cell_warnings.append(warn)
                self.warnings.append(warn)
                print(
                    f"[scitex-notebook] WARN cell {self._exec_index}: "
                    f"out-of-order (execution_count={ec})",
                    file=sys.stderr,
                )

            # Only register name definitions if the cell ran cleanly —
            # otherwise the assignment never happened and a downstream
            # cell that reads the name should be flagged as a leak.
            if not had_error:
                for name in self._cell_stores:
                    self._name_defs[name] = (
                        self._exec_index,
                        self._tracker.session_id,
                    )
        finally:
            set_tracker(None)
            self._prev_session = self._tracker.session_id
            self._tracker = None
            self._cell_t0 = None
            self._cell_src_hash = None
            self._cell_loads = set()
            self._cell_stores = set()


# ---------------------------------------------------------------------------
# IPython extension entry point
# ---------------------------------------------------------------------------

_INSTALLED: Optional[ScitexNotebookMagics] = None


def load_ipython_extension(ipython) -> None:
    """``%load_ext scitex_notebook`` enters here.

    We attach the per-cell hook installer to ``ipython.user_ns`` under a
    private name so the user can introspect it (``_scitex_nb_magic``) but
    isn't tempted to call its internals.
    """
    global _INSTALLED
    if _INSTALLED is not None:
        return  # idempotent
    _INSTALLED = ScitexNotebookMagics(ipython)
    ipython.user_ns["_scitex_nb_magic"] = _INSTALLED
    print(
        "[scitex-notebook] cell-level Clew tracking enabled "
        f"(notebook: {_INSTALLED.notebook_path or 'unknown'})"
    )


def unload_ipython_extension(ipython) -> None:
    """``%unload_ext scitex_notebook``."""
    global _INSTALLED
    if _INSTALLED is None:
        return
    try:
        ipython.events.unregister("pre_run_cell", _INSTALLED._pre_run_cell)
        ipython.events.unregister("post_run_cell", _INSTALLED._post_run_cell)
    except Exception:
        pass
    ipython.user_ns.pop("_scitex_nb_magic", None)
    _INSTALLED = None


__all__ = ["load_ipython_extension", "unload_ipython_extension"]

# EOF
