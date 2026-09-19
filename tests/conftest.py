"""Shared pytest fixtures and setup.

Pins the working directory to the repository root so relative paths
like ``Path("data/raw")`` resolve regardless of where pytest is
invoked from.
"""

import os
from pathlib import Path

from solder_reliability import find_repo_root

os.chdir(find_repo_root(Path(__file__).parent))