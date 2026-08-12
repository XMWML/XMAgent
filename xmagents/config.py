"""Process-level paths and non-secret application defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class AppPaths:
    root: Path
    data: Path
    database: Path
    workspaces: Path
    knowledge: Path
    runtime: Path

    @classmethod
    def from_root(cls, root: Path | str = ".", data_dir: Path | str | None = None) -> "AppPaths":
        root_path = Path(root).resolve()
        data_path = Path(data_dir or os.getenv("XMAGENTS_DATA_DIR", DEFAULT_DATA_DIR))
        if not data_path.is_absolute():
            data_path = root_path / data_path
        return cls(
            root=root_path,
            data=data_path,
            database=data_path / "xmagents.sqlite3",
            workspaces=data_path / "workspaces",
            knowledge=data_path / "knowledge",
            runtime=data_path / "runtime",
        )

    def ensure(self) -> None:
        for path in (self.data, self.workspaces, self.knowledge, self.runtime):
            path.mkdir(parents=True, exist_ok=True)
            try:
                path.chmod(0o700)
            except OSError:
                pass
