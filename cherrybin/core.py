"""
cherrybin core library.

A cherrybin archive is a single SQLite file storing content-addressed
blobs plus a mapping of "benchmark name -> list of (relpath, blob hash)".
Benchmarks that share files reference the same blob, so shared data is
stored exactly once regardless of how many benchmarks use it.

Design constraints this addresses:
  * The archive lives on a shared network filesystem that handles a
    single large file well, but handles many small files poorly.
  * Reads happen far more often than writes, and writes are single
    writer (never concurrent).
  * The benchmark suite evolves over time: files get added, changed,
    or removed for a given benchmark.

To keep this safe on a network filesystem without relying on file
locking (which is unreliable over NFS/SMB), the workflow is:
  1. Build/update the archive locally (fast local disk, real locking).
  2. Publish it to the shared drive as a new, immutable, versioned
     file, then atomically flip a "current" pointer. Readers only
     ever open a fully-written file, so no read/write race is
     possible without any lock at all.
  3. Check out a benchmark's files by reading (read-only, immutable
     mode) from the shared archive, caching blobs locally by hash so
     repeated or overlapping checkouts don't re-read data you already
     have.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import socket
import sqlite3
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

__version__ = "0.1.0"
__author__ = "Delaunay"
__copyright__ = "2026, Delaunay"


def _log(msg: str) -> None:
    """Progress line for operations (locking, copying, hashing) that have
    no other feedback and can silently run for a long time."""
    print(f"[cherrybin] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def resolve_blob_file(db_path: str) -> str:
    """Append-only payload file beside the sqlite ledger (``<db>.blobs``)."""
    return os.path.abspath(db_path) + ".blobs"


def _format_size(nbytes: int) -> str:
    if nbytes >= 1_000_000_000:
        return f"{nbytes / 1e9:.1f} GB"
    if nbytes >= 1_000_000:
        return f"{nbytes / 1e6:.1f} MB"
    if nbytes >= 1_000:
        return f"{nbytes / 1e3:.1f} KB"
    return f"{nbytes} B"


def _archive_payload_size(db_path: str) -> int:
    blob = resolve_blob_file(db_path)
    return os.path.getsize(blob) if os.path.isfile(blob) else 0


def _log_archive_sizes(db_path: str, *, prefix: str) -> None:
    ledger = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    payload = _archive_payload_size(db_path)
    _log(
        f"{prefix} ledger {_format_size(ledger)}, "
        f"blob {_format_size(payload)} ({resolve_blob_file(db_path)})"
    )


__descr__ = (
    "Sqlite-backed content-addressed archive for selective checkout "
    "of large shared benchmark datasets"
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (
    hash TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    data BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_files (
    benchmark TEXT NOT NULL,
    relpath   TEXT NOT NULL,
    hash      TEXT NOT NULL REFERENCES blobs(hash),
    mtime     REAL,
    PRIMARY KEY (benchmark, relpath)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_files_benchmark
    ON benchmark_files(benchmark);

CREATE INDEX IF NOT EXISTS idx_benchmark_files_hash
    ON benchmark_files(hash);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

# Default stream size for hashing / blob copy. Override via ``io_chunk=``
# on the public APIs or ``--io-chunk`` on the CLI (bytes).
DEFAULT_IO_CHUNK = 4 * 1024 * 1024
_HASH_CHUNK = DEFAULT_IO_CHUNK

_io_chunk: ContextVar[int | None] = ContextVar("cherrybin_io_chunk", default=None)


def _current_io_chunk() -> int:
    override = _io_chunk.get()
    return override if override is not None else _HASH_CHUNK


@contextlib.contextmanager
def using_io_chunk(size: int | None):
    """Temporarily use ``size``-byte stream reads. ``None`` keeps the default."""
    if size is None:
        yield
        return
    if size < 1:
        raise ValueError(f"io_chunk must be >= 1, got {size}")
    token = _io_chunk.set(int(size))
    try:
        yield
    finally:
        _io_chunk.reset(token)


@dataclass
class IoStats:
    """Payload bytes moved to/from the archive during one operation."""

    read_bytes: int = 0
    write_bytes: int = 0
    read_seconds: float = 0.0
    write_seconds: float = 0.0

    @property
    def read_mbps(self) -> float:
        if self.read_seconds <= 0:
            return 0.0
        return self.read_bytes / 1e6 / self.read_seconds

    @property
    def write_mbps(self) -> float:
        if self.write_seconds <= 0:
            return 0.0
        return self.write_bytes / 1e6 / self.write_seconds

    def summary(self) -> str:
        parts: list[str] = []
        if self.write_bytes:
            parts.append(
                f"write {self.write_bytes / 1e6:.1f} MB @ {self.write_mbps:.0f} MB/s"
            )
        if self.read_bytes:
            parts.append(
                f"read {self.read_bytes / 1e6:.1f} MB @ {self.read_mbps:.0f} MB/s"
            )
        return ", ".join(parts) if parts else "no payload I/O"


_io_stats: ContextVar[IoStats | None] = ContextVar("cherrybin_io_stats", default=None)


def _record_io(op: Literal["read", "write"], nbytes: int, elapsed: float) -> None:
    stats = _io_stats.get()
    if stats is None or nbytes <= 0 or elapsed < 0:
        return
    if op == "read":
        stats.read_bytes += nbytes
        stats.read_seconds += elapsed
    else:
        stats.write_bytes += nbytes
        stats.write_seconds += elapsed


@contextlib.contextmanager
def tracking_io_stats():
    """Track archive payload read/write throughput for the wrapped operation."""
    stats = IoStats()
    token = _io_stats.set(stats)
    try:
        yield stats
    finally:
        _io_stats.reset(token)


@dataclass
class BenchmarkStats:
    name: str
    file_count: int
    total_bytes: int
    io: IoStats = field(default_factory=IoStats)


@dataclass
class IndexStats:
    """Result of an incremental per-benchmark index."""

    name: str
    added: int
    removed: int
    unchanged: int
    new_bytes: int
    file_count: int
    deduped: int = 0
    io: IoStats = field(default_factory=IoStats)

    @property
    def changed(self) -> bool:
        return self.added > 0 or self.removed > 0


def sha256_file(path: str, *, io_chunk: int | None = None) -> str:
    with using_io_chunk(io_chunk):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(_current_io_chunk())
                if not b:
                    break
                h.update(b)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Building / updating (always run against a LOCAL copy of the db)
# ---------------------------------------------------------------------------

def connect_writable(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    cols = {row[1] for row in con.execute("PRAGMA table_info(blobs)")}
    if "offset" not in cols:
        con.execute("ALTER TABLE blobs ADD COLUMN offset INTEGER")
    return con


def add_benchmark(
    con: sqlite3.Connection,
    source_root: str,
    benchmark: str,
    *,
    io_chunk: int | None = None,
    db_path: str = "",
) -> BenchmarkStats:
    """(Re)index one benchmark's files from <source_root>/<benchmark>/**.

    Payloads are appended to ``resolve_blob_file(db_path)``; sqlite is the ledger.
    """
    if not db_path:
        raise ValueError("db_path is required")
    with using_io_chunk(io_chunk):
        bench_dir = os.path.join(source_root, benchmark)
        if not os.path.isdir(bench_dir):
            raise FileNotFoundError(bench_dir)

        with tracking_io_stats() as io:
            # Drop this benchmark's old file list; blobs are untouched since
            # other benchmarks may still reference them.
            con.execute("DELETE FROM benchmark_files WHERE benchmark = ?", (benchmark,))

            n_files = 0
            n_new_blobs = 0
            new_bytes = 0

            for root, _, files in os.walk(bench_dir):
                for fname in files:
                    full_path = os.path.join(root, fname)
                    relpath = os.path.relpath(full_path, bench_dir).replace(os.sep, "/")
                    digest = sha256_file(full_path)
                    size = os.path.getsize(full_path)
                    mtime = os.path.getmtime(full_path)

                    if _ensure_blob(con, digest, full_path, size, db_path=db_path):
                        n_new_blobs += 1
                        new_bytes += size

                    con.execute(
                        "INSERT INTO benchmark_files (benchmark, relpath, hash, mtime) "
                        "VALUES (?, ?, ?, ?)",
                        (benchmark, relpath, digest, mtime),
                    )
                    n_files += 1

            con.commit()
            return BenchmarkStats(
                name=benchmark, file_count=n_files, total_bytes=new_bytes, io=io
            )


def _prefixed_relpath(full_path: str, root: str, prefix: str) -> str:
    relpath = os.path.relpath(full_path, root).replace(os.sep, "/")
    prefix = prefix.strip("/")
    if prefix:
        return f"{prefix}/{relpath}"
    return relpath


def _walk_roots(roots: list[tuple[str, str]]) -> dict[str, tuple[str, float, str, int]]:
    """Return {relpath: (digest, mtime, full_path, size)} for every file under roots."""
    found: dict[str, tuple[str, float, str, int]] = {}
    start = time.time()
    last_log = start
    n_files = 0
    n_bytes = 0
    for abs_dir, prefix in roots:
        if not os.path.isdir(abs_dir):
            continue
        for root, _, files in os.walk(abs_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                relpath = _prefixed_relpath(full_path, abs_dir, prefix)
                size = os.path.getsize(full_path)
                found[relpath] = (
                    sha256_file(full_path),
                    os.path.getmtime(full_path),
                    full_path,
                    size,
                )
                n_files += 1
                n_bytes += size
                now = time.time()
                if now - last_log >= 30:
                    _log(
                        f"hashing... {n_files} files, {n_bytes / 1e9:.1f} GB, "
                        f"{now - start:.0f}s elapsed (current: {relpath})"
                    )
                    last_log = now
    if n_files:
        _log(f"hashed {n_files} files, {n_bytes / 1e9:.1f} GB in {time.time() - start:.0f}s")
    return found


def _copy_file_to_blob(src, dest, size: int | None = None) -> int:
    """Copy ``src`` to ``dest`` in chunks. Both are file-like. Returns bytes copied."""
    copied = 0
    remaining = size
    while remaining is None or remaining > 0:
        step = _current_io_chunk()
        n = step if remaining is None else min(step, remaining)
        chunk = src.read(n)
        if not chunk:
            break
        dest.write(chunk)
        copied += len(chunk)
        if remaining is not None:
            remaining -= len(chunk)
    return copied


def _stream_copy(
    src, dest, size: int | None = None, *, op: Literal["read", "write"]
) -> int:
    """Like ``_copy_file_to_blob`` but records archive payload throughput."""
    t0 = time.perf_counter()
    copied = _copy_file_to_blob(src, dest, size)
    _record_io(op, copied, time.perf_counter() - t0)
    return copied


def _blob_row(con: sqlite3.Connection, digest: str) -> tuple[int, int] | None:
    """Return ``(size, offset)`` for one blob hash, or ``None``."""
    row = con.execute(
        "SELECT size, offset FROM blobs WHERE hash = ?", (digest,)
    ).fetchone()
    if row is None:
        return None
    size, offset = row
    if offset is None:
        raise KeyError(f"blob {digest} has no offset")
    return size, offset


def _extract_range_to_file(blob_path: str, offset: int, size: int, dest_path: str) -> None:
    """Copy ``size`` bytes at ``offset`` from the append-only blob file."""
    tmp_path = dest_path + ".tmp"
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(blob_path, "rb") as src, open(tmp_path, "wb") as dest:
        src.seek(offset)
        _stream_copy(src, dest, size, op="read")
    os.replace(tmp_path, dest_path)


def _extract_file_to_path(
    con: sqlite3.Connection, digest: str, dest_path: str, db_path: str
) -> None:
    """Read one payload from the append-only ``<db>.blobs`` file."""
    row = _blob_row(con, digest)
    if row is None:
        raise KeyError(f"blob {digest} not found")
    size, offset = row
    _extract_range_to_file(resolve_blob_file(db_path), offset, size, dest_path)


def _publish_blob_file(src: str, dest: str) -> None:
    """Atomically place one immutable, content-addressed file at ``dest``.

    A no-op if ``dest`` already exists: since the path is derived from
    the content's hash, an existing file there is already byte-identical.
    Hardlinked when possible (same filesystem, e.g. both under the same
    node-local /tmp), otherwise copied; either way the final ``os.replace``
    means a crash mid-write leaves either nothing or a complete file at
    ``dest``, never a partial one a reader could pick up.
    """
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.uploading.{os.getpid()}"
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _append_file_to_blob(full_path: str, size: int, blob_path: str) -> int:
    """Append ``full_path`` to ``blob_path`` and return the byte offset."""
    offset = os.path.getsize(blob_path) if os.path.exists(blob_path) else 0
    with open(full_path, "rb") as src, open(blob_path, "ab") as dest:
        _stream_copy(src, dest, size, op="write")
    return offset


def _insert_appended_file(
    con: sqlite3.Connection, digest: str, full_path: str, size: int, db_path: str
) -> int:
    """Append one file to the archive blob and record it in the sqlite ledger."""
    offset = _append_file_to_blob(full_path, size, resolve_blob_file(db_path))
    con.execute(
        "INSERT INTO blobs (hash, size, data, offset) VALUES (?, ?, zeroblob(0), ?)",
        (digest, size, offset),
    )
    return size


def _ensure_blob(
    con: sqlite3.Connection,
    digest: str,
    full_path: str,
    size: int,
    *,
    db_path: str,
) -> int:
    """Insert the file if missing. Returns bytes of new blob data written."""
    if con.execute("SELECT 1 FROM blobs WHERE hash = ?", (digest,)).fetchone() is not None:
        return 0
    return _insert_appended_file(con, digest, full_path, size, db_path)


def index_roots(
    con: sqlite3.Connection,
    name: str,
    roots: list[tuple[str, str]],
    *,
    io_chunk: int | None = None,
    db_path: str = "",
) -> IndexStats:
    """Incrementally sync one benchmark's file list from ``roots``.

    Payloads append to ``resolve_blob_file(db_path)``; sqlite is the ledger.
    """
    if not db_path:
        raise ValueError("db_path is required")
    with using_io_chunk(io_chunk):
        return _index_roots(con, name, roots, db_path)


def _index_roots(
    con: sqlite3.Connection,
    name: str,
    roots: list[tuple[str, str]],
    db_path: str,
) -> IndexStats:
    old = {
        relpath: digest
        for relpath, digest in con.execute(
            "SELECT relpath, hash FROM benchmark_files WHERE benchmark = ?",
            (name,),
        )
    }
    with tracking_io_stats() as io:
        new = _walk_roots(roots)

        added = 0
        removed = 0
        unchanged = 0
        deduped = 0
        new_bytes = 0
        start = time.time()
        last_log = start

        for relpath, (digest, mtime, full_path, size) in new.items():
            old_hash = old.get(relpath)
            if old_hash == digest:
                unchanged += 1
                continue

            written = _ensure_blob(con, digest, full_path, size, db_path=db_path)
            new_bytes += written
            if written == 0:
                deduped += 1
            if old_hash is None:
                con.execute(
                    "INSERT INTO benchmark_files (benchmark, relpath, hash, mtime) "
                    "VALUES (?, ?, ?, ?)",
                    (name, relpath, digest, mtime),
                )
            else:
                con.execute(
                    "UPDATE benchmark_files SET hash = ?, mtime = ? "
                    "WHERE benchmark = ? AND relpath = ?",
                    (digest, mtime, name, relpath),
                )
            added += 1
            # Commit per file rather than once for the whole benchmark: bounds
            # how much WAL a huge tree piles up before it can checkpoint, and
            # means a crash mid-benchmark only loses the file in flight
            # instead of every file already stored.
            con.commit()
            now = time.time()
            if now - last_log >= 30:
                _log(
                    f"'{name}': {added} files committed, {new_bytes / 1e9:.1f} GB new, "
                    f"{now - start:.0f}s elapsed"
                )
                last_log = now

        for relpath in old:
            if relpath not in new:
                con.execute(
                    "DELETE FROM benchmark_files WHERE benchmark = ? AND relpath = ?",
                    (name, relpath),
                )
                removed += 1

        con.commit()
        if added or removed:
            _log(
                f"'{name}': {added} file entries, "
                f"{new_bytes / 1e9:.1f} GB new blobs, {deduped} deduped"
            )
        return IndexStats(
            name=name,
            added=added,
            removed=removed,
            unchanged=unchanged,
            new_bytes=new_bytes,
            file_count=len(new),
            deduped=deduped,
            io=io,
        )


def remove_benchmark(con: sqlite3.Connection, benchmark: str) -> int:
    cur = con.execute("DELETE FROM benchmark_files WHERE benchmark = ?", (benchmark,))
    con.commit()
    return cur.rowcount


def gc_unreferenced_blobs(con: sqlite3.Connection) -> int:
    """Remove unreferenced blobs from the sqlite ledger, then VACUUM.

    The append-only ``<db>.blobs`` file is not rewritten; dead regions remain
    until a future defrag command compacts it.
    """
    cur = con.execute(
        "DELETE FROM blobs WHERE hash NOT IN (SELECT hash FROM benchmark_files)"
    )
    con.commit()
    removed = cur.rowcount
    con.execute("VACUUM")
    return removed


def list_benchmarks(con: sqlite3.Connection) -> list[BenchmarkStats]:
    rows = con.execute(
        "SELECT benchmark, COUNT(*), COALESCE(SUM(size), 0) "
        "FROM benchmark_files JOIN blobs USING(hash) "
        "GROUP BY benchmark ORDER BY benchmark"
    ).fetchall()
    return [BenchmarkStats(name=n, file_count=c, total_bytes=s) for n, c, s in rows]


# ---------------------------------------------------------------------------
# Publishing (local db -> shared drive, atomically)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def publish_lock(shared_dir: str, timeout: float = 600):
    """
    Advisory lock so two writers can't publish at the same time.

    Uses O_CREAT|O_EXCL (atomic create-if-absent) instead of flock/fcntl,
    since real file locks are unreliable over NFS/SMB. Readers never
    need this lock - the versioned-file + atomic-pointer scheme below
    already makes concurrent reads safe with no locking at all.
    """
    os.makedirs(shared_dir, exist_ok=True)
    with publish_lock_path(os.path.join(shared_dir, ".publish.lock"), timeout=timeout):
        yield


@contextlib.contextmanager
def file_lock(db_path: str, timeout: float = 600):
    """Advisory lock beside a shared ``.db`` file (``<db>.publish.lock``)."""
    parent = os.path.dirname(os.path.abspath(db_path)) or "."
    os.makedirs(parent, exist_ok=True)
    with publish_lock_path(os.path.abspath(db_path) + ".publish.lock", timeout=timeout):
        yield


@contextlib.contextmanager
def publish_lock_path(lock_path: str, timeout: float = 600):
    """O_CREAT|O_EXCL lock at an explicit path."""
    info = f"{socket.gethostname()} pid={os.getpid()} at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    start = time.time()
    waited = False
    last_log = start
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, info.encode())
            os.close(fd)
            break
        except FileExistsError:
            elapsed = time.time() - start
            if elapsed > timeout:
                try:
                    holder = open(lock_path).read().strip()
                except OSError:
                    holder = "unknown"
                raise TimeoutError(
                    f"publish lock held by [{holder}] and timeout ({timeout}s) "
                    f"exceeded; remove {lock_path} manually if that process is dead"
                )
            if not waited:
                try:
                    holder = open(lock_path).read().strip()
                except OSError:
                    holder = "unknown"
                _log(f"waiting for lock {lock_path} held by [{holder}]")
                waited = True
            elif time.time() - last_log >= 30:
                _log(f"still waiting for lock {lock_path} ({elapsed:.0f}s elapsed)")
                last_log = time.time()
            time.sleep(2)

    if waited:
        _log(f"acquired lock {lock_path} after {time.time() - start:.0f}s")

    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def publish(local_path: str, shared_dir: str, version: str) -> str:
    """Copy local_path to shared_dir as a new immutable version, flip pointer.

    Big file bytes live in ``resolve_blob_file(local_path)``, not inside
    the .db -- copy that append-only blob beside the published archive
    before flipping the version pointer.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)

    os.makedirs(shared_dir, exist_ok=True)
    final_name = f"archive_{version}.db"
    final_path = os.path.join(shared_dir, final_name)
    tmp_path = final_path + ".uploading"

    if os.path.exists(final_path):
        raise FileExistsError(f"{final_path} already exists, choose a new version name")

    local_blob = resolve_blob_file(local_path)
    if os.path.isfile(local_blob):
        _publish_blob_file(local_blob, resolve_blob_file(final_path))

    shutil.copyfile(local_path, tmp_path)
    os.replace(tmp_path, final_path)  # atomic: readers never see a partial file

    pointer_path = os.path.join(shared_dir, "current.txt")
    pointer_tmp = pointer_path + ".tmp"
    with open(pointer_tmp, "w") as f:
        f.write(final_name + "\n")
    os.replace(pointer_tmp, pointer_path)  # atomic pointer flip

    return final_path


def resolve_current(shared_dir: str) -> str:
    """Read the 'current' pointer and return the full path to the active archive."""
    pointer_path = os.path.join(shared_dir, "current.txt")
    with open(pointer_path) as f:
        name = f.read().strip()
    return os.path.join(shared_dir, name)


def _checkpoint_and_close(con: sqlite3.Connection) -> None:
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()


def update_files(
    shared_db: str,
    items: list[tuple[str, list[tuple[str, str]]]],
    *,
    lock_timeout: float = 600.0,
    io_chunk: int | None = None,
) -> list[IndexStats]:
    """Create or incrementally update ``shared_db`` from named root lists.

    Updates the sqlite ledger and blob file in place under an advisory lock.
    Payload bytes (see ``_ensure_blob``) are appended to
    ``resolve_blob_file(shared_db)`` as each file is indexed. A crash
    mid-index may leave dead tail bytes in the blob file until defrag;
    re-running the update retries ledger writes only.
    """
    shared_db = os.path.abspath(shared_db)

    with file_lock(shared_db, timeout=lock_timeout):
        if os.path.exists(shared_db):
            _log_archive_sizes(shared_db, prefix="opening")

        con = connect_writable(shared_db)
        stats = []
        try:
            for name, roots in items:
                _log(f"indexing '{name}'...")
                t0 = time.time()
                stat = index_roots(
                    con, name, roots, io_chunk=io_chunk, db_path=shared_db
                )
                stats.append(stat)
                _log(
                    f"indexed '{name}' in {time.time() - t0:.0f}s, "
                    f"{_format_size(stat.new_bytes)} new payload, "
                    f"{stat.deduped} deduped"
                )
                _log_archive_sizes(shared_db, prefix="archive now")
        finally:
            _checkpoint_and_close(con)

    return stats


def update_file(
    shared_db: str,
    name: str,
    roots: list[tuple[str, str]],
    *,
    lock_timeout: float = 600.0,
    io_chunk: int | None = None,
) -> IndexStats:
    """Create or incrementally update one benchmark in ``shared_db``."""
    return update_files(
        shared_db, [(name, roots)], lock_timeout=lock_timeout, io_chunk=io_chunk
    )[0]


def list_files(db_path: str, benchmark: str) -> list[tuple[str, str]]:
    """Return ``[(relpath, hash), ...]`` for one benchmark."""
    con = open_readonly(db_path)
    try:
        return con.execute(
            "SELECT relpath, hash FROM benchmark_files WHERE benchmark = ? ORDER BY relpath",
            (benchmark,),
        ).fetchall()
    finally:
        con.close()


def link_or_copy(src: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest):
        os.remove(dest)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


# ---------------------------------------------------------------------------
# Checkout (read-only, safe against a live shared file)
# ---------------------------------------------------------------------------

def open_readonly(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    # immutable=1 tells sqlite the file won't change during this connection,
    # which skips locking machinery entirely - important on a network FS.
    uri = f"file:{db_path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


@dataclass
class CheckoutResult:
    benchmark: str
    dest: str
    file_count: int
    pulled_from_archive: int
    already_cached: int
    io: IoStats = field(default_factory=IoStats)


def checkout(
    db_path: str,
    benchmark: str,
    dest: str,
    cache_dir: str,
    *,
    io_chunk: int | None = None,
) -> CheckoutResult:
    with using_io_chunk(io_chunk):
        with tracking_io_stats() as io:
            result = _checkout(db_path, benchmark, dest, cache_dir)
            result.io = io
            return result


def _checkout(db_path: str, benchmark: str, dest: str, cache_dir: str) -> CheckoutResult:
    con = open_readonly(db_path)
    try:
        rows = con.execute(
            "SELECT relpath, hash FROM benchmark_files WHERE benchmark = ? ORDER BY hash",
            (benchmark,),
        ).fetchall()

        if not rows:
            raise KeyError(f"no files found for benchmark '{benchmark}' in {db_path}")

        os.makedirs(dest, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        n_from_cache = 0
        n_extracted = 0

        for relpath, digest in rows:
            cache_path = os.path.join(cache_dir, digest[:2], digest)
            dest_path = os.path.join(dest, relpath)
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

            if not os.path.exists(cache_path):
                _extract_file_to_path(con, digest, cache_path, db_path=db_path)
                n_extracted += 1
            else:
                n_from_cache += 1

            link_or_copy(cache_path, dest_path)

        return CheckoutResult(
            benchmark=benchmark,
            dest=dest,
            file_count=len(rows),
            pulled_from_archive=n_extracted,
            already_cached=n_from_cache,
        )
    finally:
        con.close()
