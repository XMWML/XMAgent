"""Safe imported SQLite knowledge bases with an explicit documents contract."""

from __future__ import annotations

import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .database import utcnow
from .files import MAX_FILE_BYTES


# Keep this in one ordered tuple so error messages, tests and the documented
# import contract cannot drift from one another. Extra application-specific
# columns are allowed, but all five documented columns must be present.
DOCUMENT_COLUMNS = ("id", "title", "content", "metadata_json", "updated_at")


class KnowledgeError(ValueError):
    pass


class KnowledgeService:
    def __init__(self, service: Any):
        self.service = service

    def import_database(self, source: str | Path, name: str) -> dict[str, Any]:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise KnowledgeError("知识库文件不存在")
        if source_path.stat().st_size > MAX_FILE_BYTES:
            raise KnowledgeError(f"知识库文件超过 {MAX_FILE_BYTES // 1024 // 1024}MB 限制")
        if not name.strip():
            raise KnowledgeError("知识库名称不能为空")

        destination: Path | None = None
        index_path: Path | None = None
        uri = f"file:{source_path.as_posix()}?mode=ro"
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(uri, uri=True)
            with connection:
                self._validate_documents_schema(connection)
                count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        except sqlite3.Error as error:
            raise KnowledgeError(f"无法读取 SQLite 知识库：{error}") from error
        except KnowledgeError:
            raise
        finally:
            if connection is not None:
                connection.close()

        knowledge_id = uuid.uuid4().hex
        try:
            destination = self.service.paths.knowledge / f"{knowledge_id}.sqlite3"
            shutil.copy2(source_path, destination)
            try:
                destination.chmod(0o600)
            except OSError:
                pass
            # Build a read-only FTS sidecar under our control.  The original
            # import stays intact and can be inspected/exported; searching
            # never needs to execute writes against user supplied SQLite files.
            index_path = self._index_path(knowledge_id)
            self._build_fts_index(destination, index_path)
            now = utcnow()
            self.service.db.execute("INSERT INTO knowledge_bases(id,name,source_path,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                                    (knowledge_id, name.strip(), str(destination), 1, now, now))
            self.service.db.audit("knowledge_imported", target=knowledge_id, detail={"documents": count})
            return {"id": knowledge_id, "name": name.strip(), "documents": count, "path": str(destination), "index_path": str(index_path)}
        except Exception:
            # A failed index or database write must not leave an inaccessible
            # imported copy behind. The caller's source file is never touched.
            if index_path is not None:
                index_path.unlink(missing_ok=True)
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise

    @staticmethod
    def _validate_documents_schema(connection: sqlite3.Connection) -> None:
        """Validate the documented read-only import schema before copying it."""

        rows = connection.execute("PRAGMA table_info(documents)").fetchall()
        by_name = {str(row[1]): row for row in rows}
        missing = [column for column in DOCUMENT_COLUMNS if column not in by_name]
        if missing:
            expected = ", ".join(DOCUMENT_COLUMNS)
            raise KnowledgeError(f"知识库必须包含 documents({expected}) 表")

        # ``id`` is the stable search result identity and must be declared as a
        # primary key. ``content`` is what we index, so accepting nullable
        # content would create a knowledge base that cannot meet its contract.
        if int(by_name["id"][5]) <= 0:
            raise KnowledgeError("知识库 documents.id 必须是 PRIMARY KEY")
        if int(by_name["content"][3]) != 1:
            raise KnowledgeError("知识库 documents.content 必须声明为 NOT NULL")
        if not all("TEXT" in str(by_name[column][2] or "").upper() for column in DOCUMENT_COLUMNS):
            raise KnowledgeError("知识库 documents 的 id、title、content、metadata_json、updated_at 必须为 TEXT 类型")

    def _index_path(self, knowledge_id: str) -> Path:
        return self.service.paths.knowledge / f"{knowledge_id}.fts.sqlite3"

    @staticmethod
    def _build_fts_index(source: Path, index_path: Path) -> None:
        index_path.unlink(missing_ok=True)
        origin = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
        index = sqlite3.connect(index_path)
        try:
            with origin, index:
                index.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(id UNINDEXED, title, content, metadata_json UNINDEXED)")
                rows = origin.execute("SELECT id, coalesce(title, ''), content, coalesce(metadata_json, '{}') FROM documents")
                index.executemany("INSERT INTO documents_fts(id,title,content,metadata_json) VALUES(?,?,?,?)", rows)
        finally:
            origin.close()
            index.close()
        try:
            index_path.chmod(0o600)
        except OSError:
            pass

    def _path(self, knowledge_id: str) -> Path:
        row = self.service.db.fetchone("SELECT source_path FROM knowledge_bases WHERE id=? AND enabled=1", (knowledge_id,))
        if not row:
            raise KnowledgeError("知识库不存在或已禁用")
        return Path(row["source_path"])

    def schema(self, knowledge_id: str) -> dict[str, Any]:
        path = self._path(knowledge_id)
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            columns = [row[1] for row in connection.execute("PRAGMA table_info(documents)")]
            count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        finally:
            connection.close()
        return {"table": "documents", "columns": columns, "count": count}

    def search(self, knowledge_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        path = self._path(knowledge_id)
        terms = [term for term in query.replace("'", " ").split() if term][:6]
        if not terms:
            return []
        index_path = self._index_path(knowledge_id)
        if index_path.exists():
            # FTS5 supports quoted tokens and ranks matching passages.  Fall
            # back to portable LIKE for malformed queries or older databases.
            match = " AND ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
            try:
                connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
                try:
                    connection.row_factory = sqlite3.Row
                    rows = connection.execute(
                        "SELECT id,title,substr(content,1,4000) AS content,metadata_json FROM documents_fts WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT ?",
                        (match, max(1, min(limit, 20))),
                    ).fetchall()
                finally:
                    connection.close()
                if rows:
                    return [dict(row) for row in rows]
            except sqlite3.Error:
                pass
        predicate = " AND ".join("lower(content || ' ' || coalesce(title,'')) LIKE ?" for _ in terms)
        args = tuple(f"%{term.lower()}%" for term in terms) + (max(1, min(limit, 20)),)
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT id, coalesce(title, '') AS title, substr(content, 1, 4000) AS content, coalesce(metadata_json, '{{}}') AS metadata_json FROM documents WHERE {predicate} LIMIT ?", args
            ).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in rows]
