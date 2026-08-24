"""Tests for cherrybin.core: build -> publish -> checkout round trip."""

import os

from cherrybin.core import (
    add_benchmark,
    checkout,
    connect_writable,
    gc_unreferenced_blobs,
    index_roots,
    list_benchmarks,
    list_files,
    publish,
    remove_benchmark,
    resolve_current,
    update_file,
    update_files,
)


def build_local_db(tmp_path, source_tree):
    db_path = str(tmp_path / "local.db")
    con = connect_writable(db_path)
    add_benchmark(con, source_tree, "bench_a")
    add_benchmark(con, source_tree, "bench_b")
    con.close()
    return db_path


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
    index_roots(con, "bench_a", [(str(a), "")])
    index_roots(con, "bench_b", [(str(b), "")])
    n_blobs = con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    assert n_blobs == 3

    os.remove(a / "only_a.bin")
    index_roots(con, "bench_a", [(str(a), "")])
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
