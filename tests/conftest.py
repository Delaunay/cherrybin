"""Shared test fixtures."""

import pytest


@pytest.fixture
def source_tree(tmp_path):
    """A small source tree with two benchmarks sharing one file."""
    src = tmp_path / "src"
    (src / "bench_a").mkdir(parents=True)
    (src / "bench_b").mkdir(parents=True)

    (src / "bench_a" / "common.bin").write_text("shared data")
    (src / "bench_b" / "common.bin").write_text("shared data")
    (src / "bench_a" / "a_only.bin").write_text("only in a")
    (src / "bench_b" / "b_only.bin").write_text("only in b")

    return str(src)
