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
from dataclasses import dataclass

__version__ = "0.1.0"
__author__ = "Delaunay"
__copyright__ = "2026, Delaunay"
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

_HASH_CHUNK = 4 * 1024 * 1024  # 4MB read chunks for hashing large files


@dataclass
class BenchmarkStats:
    name: str
    file_count: int
    total_bytes: int


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_HASH_CHUNK)
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
    return con


def add_benchmark(con: sqlite3.Connection, source_root: str, benchmark: str) -> BenchmarkStats:
    """(Re)index one benchmark's files from <source_root>/<benchmark>/**."""
    bench_dir = os.path.join(source_root, benchmark)
    if not os.path.isdir(bench_dir):
        raise FileNotFoundError(bench_dir)

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

            exists = con.execute("SELECT 1 FROM blobs WHERE hash = ?", (digest,)).fetchone()
            if exists is None:
                with open(full_path, "rb") as f:
                    data = f.read()
                con.execute(
                    "INSERT INTO blobs (hash, size, data) VALUES (?, ?, ?)",
                    (digest, size, data),
                )
                n_new_blobs += 1
                new_bytes += size

            con.execute(
                "INSERT INTO benchmark_files (benchmark, relpath, hash, mtime) "
                "VALUES (?, ?, ?, ?)",
                (benchmark, relpath, digest, mtime),
            )
            n_files += 1

    con.commit()
    return BenchmarkStats(name=benchmark, file_count=n_files, total_bytes=new_bytes)


def remove_benchmark(con: sqlite3.Connection, benchmark: str) -> int:
    cur = con.execute("DELETE FROM benchmark_files WHERE benchmark = ?", (benchmark,))
    con.commit()
    return cur.rowcount


def gc_unreferenced_blobs(con: sqlite3.Connection) -> int:
    """Remove blobs no longer referenced by any benchmark, then VACUUM."""
    cur = con.execute(
        "DELETE FROM blobs WHERE hash NOT IN (SELECT DISTINCT hash FROM benchmark_files)"
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
    lock_path = os.path.join(shared_dir, ".publish.lock")
    info = f"{socket.gethostname()} pid={os.getpid()} at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"

    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, info.encode())
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout:
                try:
                    holder = open(lock_path).read().strip()
                except OSError:
                    holder = "unknown"
                raise TimeoutError(
                    f"publish lock held by [{holder}] and timeout ({timeout}s) "
                    f"exceeded; remove {lock_path} manually if that process is dead"
                )
            time.sleep(2)

    try:
        yield
    finally:
        os.remove(lock_path)


def publish(local_path: str, shared_dir: str, version: str) -> str:
    """Copy local_path to shared_dir as a new immutable version, flip pointer."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)

    os.makedirs(shared_dir, exist_ok=True)
    final_name = f"archive_{version}.db"
    final_path = os.path.join(shared_dir, final_name)
    tmp_path = final_path + ".uploading"

    if os.path.exists(final_path):
        raise FileExistsError(f"{final_path} already exists, choose a new version name")

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


def checkout(db_path: str, benchmark: str, dest: str, cache_dir: str) -> CheckoutResult:
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
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                (data,) = con.execute(
                    "SELECT data FROM blobs WHERE hash = ?", (digest,)
                ).fetchone()
                tmp_path = cache_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    f.write(data)
                os.replace(tmp_path, cache_path)
                n_extracted += 1
            else:
                n_from_cache += 1

            if os.path.exists(dest_path):
                os.remove(dest_path)
            try:
                os.link(cache_path, dest_path)
            except OSError:
                shutil.copyfile(cache_path, dest_path)

        return CheckoutResult(
            benchmark=benchmark,
            dest=dest,
            file_count=len(rows),
            pulled_from_archive=n_extracted,
            already_cached=n_from_cache,
        )
    finally:
        con.close()
