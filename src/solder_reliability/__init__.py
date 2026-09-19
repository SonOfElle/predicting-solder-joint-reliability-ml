"""Solder joint reliability prediction.

Refactored from the MSc thesis:
Reliability prediction of soldering joints in electronic systems.
Silesian University of Technology, 2023.
"""

__version__ = "0.1.0"
__author__ = "Ngonidzashe Ntuli"


from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Return the repository root.

    Walks up from ``start`` (default: cwd) until it finds ``pyproject.toml``.
    Notebooks and scripts run by external tools use their own folder as cwd,
    so relative paths do not resolve from repo root. Call ``os.chdir(find_repo_root())`` at the top of every entry
    point, or use the returned path directly.
    """
    p = Path(start) if start is not None else Path.cwd()
    p = p.resolve()
    for candidate in (p, *p.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"pyproject.toml not found above {p}")