"""Workspace-only file staging and path validation."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


MAX_FILE_BYTES = 100 * 1024 * 1024


class FileSafetyError(ValueError):
    pass


def workspace_path(workspace: str | Path, relative_path: str | Path) -> Path:
    root = Path(workspace).resolve()
    target = (root / relative_path).resolve()
    if target == root or root not in target.parents:
        raise FileSafetyError("文件必须位于当前 agent 工作区内")
    return target


def stage_upload(source: str | Path, workspace: str | Path, filename: str | None = None) -> dict[str, str | int]:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileSafetyError("上传源文件不存在")
    size = source_path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise FileSafetyError(f"文件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
    clean_name = Path(filename or source_path.name).name or "attachment"
    upload_dir = workspace_path(workspace, "uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / clean_name
    counter = 1
    while target.exists():
        target = upload_dir / f"{Path(clean_name).stem}-{counter}{Path(clean_name).suffix}"
        counter += 1
    shutil.copyfile(source_path, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"path": str(target), "relative_path": str(target.relative_to(Path(workspace))), "filename": target.name, "size": size, "sha256": digest}


def validate_send_file(workspace: str | Path, relative_path: str | Path) -> Path:
    target = workspace_path(workspace, relative_path)
    if not target.is_file():
        raise FileSafetyError("指定文件不存在")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise FileSafetyError(f"文件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
    return target
