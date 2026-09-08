"""Tests for cherrybin.core: build -> publish -> checkout round trip."""

import os

import pytest

from cherrybin.core import (
    add_benchmark,
    checkout,
    checkout_benchmarks,
    connect_writable,
    gc_unreferenced_blobs,
    index_roots,
    list_benchmarks,
    list_files,
    open_readonly,
    publish,
    remove_benchmark,
    resolve_blob_file,
    resolve_current,
    update_file,
    update_files,
    using_io_chunk,
    _checkout_plan,
    _merge_contiguous_ranges,
)


def build_local_db(tmp_path, source_tree):
    db_path = str(tmp_path / "local.db")
    con = connect_writable(db_path)
    add_benchmark(con, source_tree, "bench_a", db_path=db_path)
    add_benchmark(con, source_tree, "bench_b", db_path=db_path)
    con.close()
    return db_path


def _read_blob_slice(blob_path, offset, size):
    with open(blob_path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def test_build_indexes_both_benchmarks(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    con = connect_writable(db_path)
    stats = {s.name: s for s in list_benchmarks(con)}
    con.close()

    assert set(stats) == {"bench_a", "bench_b"}
    assert stats["bench_a"].file_count == 2
    assert stats["bench_b"].file_count == 2


def test_dedup_stores_shared_blob_once(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    con = connect_writable(db_path)
    n_blobs = con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    con.close()

    # 3 distinct contents total: common.bin (shared), a_only.bin, b_only.bin
    assert n_blobs == 3


def test_checkout_streams_large_file(tmp_path, monkeypatch):
    import cherrybin.core as core

    monkeypatch.setattr(core, "_HASH_CHUNK", 16)
    src = tmp_path / "src" / "bench"
    src.mkdir(parents=True)
    payload = b"abcdefghij" * 20  # 200 bytes, many 16-byte chunks
    (src / "big.bin").write_bytes(payload)

    db_path = str(tmp_path / "local.db")
    con = connect_writable(db_path)
    add_benchmark(con, str(tmp_path / "src"), "bench", db_path=db_path)
    con.close()

    dest = str(tmp_path / "out")
    checkout(db_path, "bench", dest, str(tmp_path / "cache"))
    assert open(os.path.join(dest, "big.bin"), "rb").read() == payload


def test_io_chunk_kwarg_overrides_default(tmp_path, monkeypatch):
    import cherrybin.core as core

    seen = []
    real_read = core._current_io_chunk

    def spy():
        n = real_read()
        seen.append(n)
        return n

    monkeypatch.setattr(core, "_current_io_chunk", spy)
    src = tmp_path / "src" / "bench"
    src.mkdir(parents=True)
    (src / "a.bin").write_bytes(b"x" * 64)

    db_path = str(tmp_path / "local.db")
    con = connect_writable(db_path)
    add_benchmark(con, str(tmp_path / "src"), "bench", io_chunk=16, db_path=db_path)
    con.close()

    checkout(db_path, "bench", str(tmp_path / "out"), str(tmp_path / "cache"), io_chunk=32)
    assert 16 in seen
    assert 32 in seen
    assert open(os.path.join(tmp_path, "out", "a.bin"), "rb").read() == b"x" * 64


def test_using_io_chunk_rejects_non_positive():
    with pytest.raises(ValueError):
        with using_io_chunk(0):
            pass


def test_checkout_materializes_correct_files(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    dest = str(tmp_path / "out" / "bench_a")
    cache = str(tmp_path / "cache")

    result = checkout(db_path, "bench_a", dest, cache)

    assert result.file_count == 2
    assert os.path.exists(os.path.join(dest, "common.bin"))
    assert os.path.exists(os.path.join(dest, "a_only.bin"))
    with open(os.path.join(dest, "common.bin")) as f:
        assert f.read() == "shared data"


def test_checkout_hardlinks_shared_blob_across_benchmarks(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    cache = str(tmp_path / "cache")

    checkout(db_path, "bench_a", str(tmp_path / "out" / "bench_a"), cache)
    result_b = checkout(db_path, "bench_b", str(tmp_path / "out" / "bench_b"), cache)

    # bench_b's common.bin should come from cache, not be re-extracted
    assert result_b.already_cached == 1
    assert result_b.pulled_from_archive == 1  # only b_only.bin is new

    path_a = os.path.join(tmp_path, "out", "bench_a", "common.bin")
    path_b = os.path.join(tmp_path, "out", "bench_b", "common.bin")
    assert os.stat(path_a).st_ino == os.stat(path_b).st_ino  # same inode = hardlinked


def test_publish_and_resolve_current(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    shared_dir = str(tmp_path / "shared")

    published_path = publish(db_path, shared_dir, "v1")

    assert os.path.exists(published_path)
    assert resolve_current(shared_dir) == published_path


def test_publish_then_checkout_via_shared_dir(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    shared_dir = str(tmp_path / "shared")
    publish(db_path, shared_dir, "v1")

    current_db = resolve_current(shared_dir)
    result = checkout(current_db, "bench_a", str(tmp_path / "out"), str(tmp_path / "cache"))

    assert result.file_count == 2


def test_remove_then_gc_reclaims_unreferenced_blob(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    con = connect_writable(db_path)

    remove_benchmark(con, "bench_a")
    n_blobs_before = con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    removed = gc_unreferenced_blobs(con)
    n_blobs_after = con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    con.close()

    # a_only.bin's blob becomes unreferenced once bench_a is removed;
    # common.bin's blob is kept since bench_b still references it.
    assert removed == 1
    assert n_blobs_after == n_blobs_before - 1


def test_second_publish_gets_new_version(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    shared_dir = str(tmp_path / "shared")

    publish(db_path, shared_dir, "v1")
    publish(db_path, shared_dir, "v2")

    assert resolve_current(shared_dir).endswith("archive_v2.db")
    assert os.path.exists(os.path.join(shared_dir, "archive_v1.db"))  # old version kept


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_index_roots_prefixes_and_incremental(tmp_path):
    data = tmp_path / "data" / "vllm"
    cache = tmp_path / "cache" / "vllm"
    _write(str(data / "hub" / "model.bin"), "weights")
    _write(str(cache / "torch" / "hub.bin"), "torch")

    db_path = str(tmp_path / "local.db")
    con = connect_writable(db_path)
    stats = index_roots(
        con,
        "vllm",
        [(str(data), "data"), (str(cache), "cache")],
        db_path=db_path,
    )
    con.close()

    assert stats.added == 2
    assert stats.removed == 0
    assert stats.unchanged == 0
    assert stats.file_count == 2
    assert stats.changed

    paths = {rel for rel, _ in list_files(db_path, "vllm")}
    assert paths == {"data/hub/model.bin", "cache/torch/hub.bin"}

    # Second index with the same trees is a no-op
    con = connect_writable(db_path)
    again = index_roots(
        con,
        "vllm",
        [(str(data), "data"), (str(cache), "cache")],
        db_path=db_path,
    )
    assert again.added == 0
    assert again.removed == 0
    assert again.unchanged == 2
    assert not again.changed

    # Add one file, delete one file
    os.remove(data / "hub" / "model.bin")
    _write(str(data / "hub" / "new.bin"), "fresh")
    stats = index_roots(
        con,
        "vllm",
        [(str(data), "data"), (str(cache), "cache")],
        db_path=db_path,
    )
    con.close()

    assert stats.added == 1
    assert stats.removed == 1
    assert stats.unchanged == 1
    assert stats.changed
    paths = {rel for rel, _ in list_files(db_path, "vllm")}
    assert paths == {"data/hub/new.bin", "cache/torch/hub.bin"}


def test_index_roots_shared_blob_kept_when_one_bench_drops(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write(str(a / "shared.bin"), "same")
    _write(str(a / "only_a.bin"), "a")
    _write(str(b / "shared.bin"), "same")
    _write(str(b / "only_b.bin"), "b")

    db_path = str(tmp_path / "local.db")
    con = connect_writable(db_path)
    index_roots(con, "bench_a", [(str(a), "")], db_path=db_path)
    index_roots(con, "bench_b", [(str(b), "")], db_path=db_path)
    n_blobs = con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    assert n_blobs == 3

    os.remove(a / "only_a.bin")
    index_roots(con, "bench_a", [(str(a), "")], db_path=db_path)
    n_blobs_after = con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    con.close()

    # only_a blob is unreferenced but not deleted until gc
    assert n_blobs_after == 3
    paths_a = {rel for rel, _ in list_files(db_path, "bench_a")}
    assert paths_a == {"shared.bin"}


def test_update_file_creates_and_replaces(tmp_path):
    tree = tmp_path / "tree" / "bench"
    _write(str(tree / "a.bin"), "one")
    shared = str(tmp_path / "shared" / "archive.db")

    stats = update_file(shared, "bench", [(str(tree), "data")])
    assert stats.added == 1
    assert os.path.exists(shared)
    assert list_files(shared, "bench")[0][0] == "data/a.bin"

    inode_before = os.stat(shared).st_ino
    unchanged = update_file(shared, "bench", [(str(tree), "data")])
    assert not unchanged.changed
    assert os.stat(shared).st_ino == inode_before

    _write(str(tree / "b.bin"), "two")
    changed = update_file(shared, "bench", [(str(tree), "data")])
    assert changed.added == 1
    assert changed.unchanged == 1
    assert {rel for rel, _ in list_files(shared, "bench")} == {"data/a.bin", "data/b.bin"}

    dest = str(tmp_path / "out")
    result = checkout(shared, "bench", dest, str(tmp_path / "blob_cache"))
    assert result.file_count == 2
    assert os.path.exists(os.path.join(dest, "data", "a.bin"))


def test_append_blob_round_trip(tmp_path):
    payload_a = b"first-file-contents"
    payload_b = b"second-file-is-longer!!"
    src = tmp_path / "src" / "bench"
    src.mkdir(parents=True)
    (src / "a.bin").write_bytes(payload_a)
    (src / "b.bin").write_bytes(payload_b)

    db_path = str(tmp_path / "local.db")
    blob_file = resolve_blob_file(db_path)
    con = connect_writable(db_path)
    add_benchmark(con, str(tmp_path / "src"), "bench", db_path=db_path)

    assert os.path.isfile(blob_file)
    assert not os.path.isdir(blob_file)

    rows = con.execute(
        "SELECT hash, size, offset, length(data) FROM blobs ORDER BY offset"
    ).fetchall()
    con.close()

    assert len(rows) == 2
    assert all(length == 0 for _, _, _, length in rows)
    assert rows[0][2] == 0
    assert rows[1][2] == rows[0][1]
    by_size = {size: offset for _, size, offset, _ in rows}
    assert _read_blob_slice(blob_file, by_size[len(payload_a)], len(payload_a)) == payload_a
    assert _read_blob_slice(blob_file, by_size[len(payload_b)], len(payload_b)) == payload_b

    dest = str(tmp_path / "out")
    checkout(db_path, "bench", dest, str(tmp_path / "cache"))
    assert open(os.path.join(dest, "a.bin"), "rb").read() == payload_a
    assert open(os.path.join(dest, "b.bin"), "rb").read() == payload_b


def test_append_blob_gc_keeps_blob_file(tmp_path):
    payload = b"x" * 80
    src = tmp_path / "src" / "bench"
    src.mkdir(parents=True)
    (src / "huge.bin").write_bytes(payload)

    db_path = str(tmp_path / "local.db")
    blob_file = resolve_blob_file(db_path)
    con = connect_writable(db_path)
    add_benchmark(con, str(tmp_path / "src"), "bench", db_path=db_path)
    blob_size_before = os.path.getsize(blob_file)

    remove_benchmark(con, "bench")
    removed = gc_unreferenced_blobs(con)
    n_blobs = con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    con.close()

    assert removed >= 1
    assert n_blobs == 0
    assert os.path.exists(blob_file)
    assert os.path.getsize(blob_file) == blob_size_before


def test_append_blob_dedup_appends_once(tmp_path):
    payload = os.urandom(128)
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "shared.bin").write_bytes(payload)
    (b / "shared.bin").write_bytes(payload)

    db_path = str(tmp_path / "local.db")
    blob_file = resolve_blob_file(db_path)
    con = connect_writable(db_path)
    index_roots(con, "bench_a", [(str(a), "")], db_path=db_path)
    index_roots(con, "bench_b", [(str(b), "")], db_path=db_path)
    rows = con.execute("SELECT COUNT(*), MIN(offset) FROM blobs").fetchone()
    con.close()

    assert rows[0] == 1
    assert _read_blob_slice(blob_file, rows[1], len(payload)) == payload
    assert os.path.getsize(blob_file) == len(payload)


def test_publish_syncs_append_blob(tmp_path):
    payload = os.urandom(256)
    src = tmp_path / "src" / "bench"
    src.mkdir(parents=True)
    (src / "huge.bin").write_bytes(payload)

    db_path = str(tmp_path / "local.db")
    con = connect_writable(db_path)
    add_benchmark(con, str(tmp_path / "src"), "bench", db_path=db_path)
    offset = con.execute("SELECT offset FROM blobs").fetchone()[0]
    con.close()

    shared_dir = str(tmp_path / "shared")
    published_path = publish(db_path, shared_dir, "v1")
    published_blob = resolve_blob_file(published_path)

    assert os.path.isfile(published_blob)

    assert _read_blob_slice(published_blob, offset, len(payload)) == payload

    dest = str(tmp_path / "out")
    result = checkout(published_path, "bench", dest, str(tmp_path / "cache"))
    assert result.file_count == 1
    assert open(os.path.join(dest, "huge.bin"), "rb").read() == payload


def test_update_file_appends_to_shared_blob(tmp_path):
    payload = os.urandom(200)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "big.bin").write_bytes(payload)

    shared = str(tmp_path / "archive.db")
    stats = update_file(shared, "bench", [(str(tree), "data")])
    assert stats.added == 1

    blob_file = resolve_blob_file(shared)
    offset = connect_writable(shared).execute("SELECT offset FROM blobs").fetchone()[0]
    assert _read_blob_slice(blob_file, offset, len(payload)) == payload

    dest = str(tmp_path / "out")
    checkout(shared, "bench", dest, str(tmp_path / "cache"))
    assert open(os.path.join(dest, "data", "big.bin"), "rb").read() == payload


def test_checkout_reports_read_io_stats(tmp_path):
    payload = os.urandom(256 * 1024)
    src = tmp_path / "src" / "bench"
    src.mkdir(parents=True)
    (src / "big.bin").write_bytes(payload)

    db_path = str(tmp_path / "local.db")
    con = connect_writable(db_path)
    add_benchmark(con, str(tmp_path / "src"), "bench", db_path=db_path)
    con.close()

    result = checkout(db_path, "bench", str(tmp_path / "out"), str(tmp_path / "cache"))
    assert result.pulled_from_archive == 1
    assert result.io.read_bytes == len(payload)
    assert result.io.read_mbps > 0
    assert "read" in result.io.summary()


def test_update_file_reports_write_io_stats(tmp_path):
    payload = os.urandom(256 * 1024)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "big.bin").write_bytes(payload)

    shared = str(tmp_path / "archive.db")
    stats = update_file(shared, "bench", [(str(tree), "data")])
    assert stats.new_bytes == len(payload)
    assert stats.io.write_bytes == len(payload)
    assert stats.io.write_mbps > 0
    assert "write" in stats.io.summary()


def test_second_benchmark_dedups_shared_blobs(tmp_path):
    payload = os.urandom(128 * 1024)
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "shared.bin").write_bytes(payload)
    (b / "shared.bin").write_bytes(payload)

    shared = str(tmp_path / "archive.db")
    first = update_file(shared, "bench_a", [(str(a), "")])
    second = update_file(shared, "bench_b", [(str(b), "")])

    assert first.added == 1
    assert first.deduped == 0
    assert first.new_bytes == len(payload)
    assert second.added == 1
    assert second.deduped == 1
    assert second.new_bytes == 0
    assert second.io.write_bytes == 0

    con = connect_writable(shared)
    assert con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1
    con.close()


def test_update_files_two_benchmarks(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write(str(a / "x.bin"), "x")
    _write(str(b / "y.bin"), "y")
    shared = str(tmp_path / "archive.db")

    stats = update_files(
        shared,
        [("aa", [(str(a), "")]), ("bb", [(str(b), "")])],
    )
    assert [s.name for s in stats] == ["aa", "bb"]
    assert {s.name for s in list_benchmarks(connect_writable(shared))} == {"aa", "bb"}


def test_merge_contiguous_ranges_adjacent():
    assert _merge_contiguous_ranges([(0, 10), (10, 5)]) == [(0, 15)]


def test_merge_contiguous_ranges_gap():
    assert _merge_contiguous_ranges([(0, 10), (20, 5)]) == [(0, 10), (20, 25)]


def test_checkout_plan_sorted_by_offset(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    con = open_readonly(db_path)
    try:
        plan = _checkout_plan(con, benchmark="bench_a")
    finally:
        con.close()
    offsets = [entry.offset for entry in plan]
    assert offsets == sorted(offsets)


def test_checkout_stream_vs_naive_identical(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    stream_dest = str(tmp_path / "stream")
    naive_dest = str(tmp_path / "naive")
    checkout(db_path, "bench_a", stream_dest, str(tmp_path / "cache_s"), stream=True)
    checkout(db_path, "bench_a", naive_dest, str(tmp_path / "cache_n"), stream=False)

    for name in ("common.bin", "a_only.bin"):
        stream_path = os.path.join(stream_dest, name)
        naive_path = os.path.join(naive_dest, name)
        assert open(stream_path, "rb").read() == open(naive_path, "rb").read()


def test_checkout_benchmarks_single_pass(tmp_path, source_tree):
    db_path = build_local_db(tmp_path, source_tree)
    dest = str(tmp_path / "out")
    cache = str(tmp_path / "cache")

    results = checkout_benchmarks(
        db_path, ["bench_a", "bench_b"], dest, cache, stream=True
    )
    assert {r.benchmark for r in results} == {"bench_a", "bench_b"}
    assert results[0].chunks_read >= 1
    assert os.path.exists(os.path.join(dest, "bench_a", "common.bin"))
    assert os.path.exists(os.path.join(dest, "bench_b", "b_only.bin"))


def test_checkout_stream_uses_fewer_seeks(tmp_path, source_tree, monkeypatch):
    db_path = build_local_db(tmp_path, source_tree)
    blob_path = resolve_blob_file(db_path)
    seeks: list[int] = []
    real_open = open

    class SeekSpy:
        def __init__(self, f):
            self._f = f

        def read(self, n=-1):
            return self._f.read(n)

        def seek(self, offset, whence=0):
            seeks.append(offset)
            return self._f.seek(offset, whence)

        def fileno(self):
            return self._f.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._f.close()

    def spy_open(path, mode="r", *args, **kwargs):
        f = real_open(path, mode, *args, **kwargs)
        if os.path.abspath(path) == os.path.abspath(blob_path) and "b" in mode:
            return SeekSpy(f)
        return f

    monkeypatch.setattr("builtins.open", spy_open)

    checkout(db_path, "bench_a", str(tmp_path / "out"), str(tmp_path / "cache"), stream=True)
    stream_seeks = len(seeks)

    seeks.clear()
    checkout(
        db_path,
        "bench_b",
        str(tmp_path / "out2"),
        str(tmp_path / "cache"),
        stream=False,
    )
    naive_seeks = len(seeks)

    assert stream_seeks <= naive_seeks


def test_checkout_stream_writer_failure_does_not_deadlock(tmp_path, monkeypatch):
    """A writer-side failure must surface as an exception, not hang.

    With a 1-slot queue and a tiny io_chunk, the reader is guaranteed to
    still be pushing blocks (blocked on a full queue) at the moment the
    writer dies on the first blob -- exactly the state that used to make
    the reader's plain ``queue.put()`` block forever with no consumer
    left, hanging the whole checkout with no error ever raised.
    """
    import threading

    import cherrybin.core as core

    src = tmp_path / "src" / "bench"
    src.mkdir(parents=True)
    for i in range(5):
        (src / f"f{i}.bin").write_bytes(os.urandom(4096))

    db_path = str(tmp_path / "local.db")
    con = connect_writable(db_path)
    add_benchmark(con, str(tmp_path / "src"), "bench", db_path=db_path)
    con.close()

    con = open_readonly(db_path)
    plan = _checkout_plan(con, benchmark="bench")
    con.close()
    pending = sorted(
        (
            core.PendingBlob(hash=e.hash, offset=e.offset, size=e.size, dest_paths=[])
            for e in plan
        ),
        key=lambda p: p.offset,
    )

    monkeypatch.setattr(
        core,
        "_write_cache_atomically",
        lambda *a, **k: (_ for _ in ()).throw(OSError("simulated disk failure")),
    )

    result: dict = {}

    def run():
        try:
            core._stream_checkout_pending(
                resolve_blob_file(db_path),
                pending,
                str(tmp_path / "cache"),
                io_chunk=16,
                queue_depth=1,
            )
        except BaseException as exc:
            result["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)

    assert not t.is_alive(), "checkout streaming deadlocked instead of raising"
    assert "simulated disk failure" in str(result.get("error"))
