"""Tests for cherrybin.core: build -> publish -> checkout round trip."""

import os

from cherrybin.core import (
    add_benchmark,
    checkout,
    connect_writable,
    gc_unreferenced_blobs,
    list_benchmarks,
    publish,
    remove_benchmark,
    resolve_current,
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
