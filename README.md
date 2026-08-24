# cherrybin

[![Tests](https://github.com/Delaunay/cherrybin/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/Delaunay/cherrybin/actions/workflows/test.yml)

A SQLite-backed, content-addressed archive for benchmark datasets that:

- live on a shared network filesystem which handles one big file well
  but handles many small files poorly,
- are read far more often than they're written, with updates coming
  from a single writer at a time (never concurrent),
- are organized into named, overlapping "benchmarks" that often share
  files with each other, and
- evolve over time as the benchmark suite changes.

Instead of one giant `tar` you unpack in full every time, cherrybin
stores every file's content once (deduped by SHA-256 hash) in a single
`.db` file, and maps benchmark names to the files they need. You check
out only the benchmark(s) you're about to run, and shared files are
never duplicated on disk, in the archive, or over the network.

```bash
pip install -e .
```

## Concepts

- **Archive**: a single SQLite file (`blobs` table + `benchmark_files`
  table). Deliberately just one file, so copying/transferring it over
  a network filesystem is one big sequential operation instead of many
  small ones.
- **Build**: done locally, against a fast local disk with reliable
  file locking. Scans a source tree laid out as
  `<source>/<benchmark_name>/<relative/path>`, hashes each file, and
  only stores blobs that aren't already present.
- **Publish**: copies the finished local archive to the shared drive
  under a new versioned filename (`archive_v42.db`) and atomically
  flips a `current.txt` pointer. Readers only ever open a fully-written
  file — there's no window where a reader could see a half-written
  archive, so **no file lock is needed for reads**. An advisory lock
  (`.publish.lock`, using atomic `O_CREAT|O_EXCL` rather than
  `flock`/`fcntl`, since real file locks are unreliable over NFS/SMB)
  only guards against two people publishing at the same time.
- **Checkout**: opens the shared archive read-only/immutable and
  materializes one benchmark's files into a local directory. A local
  hash-keyed blob cache means files shared across benchmarks, or
  unchanged between runs, are hardlinked rather than re-read from the
  archive.

## Quick start

```bash
# Build a local archive from a source tree laid out as
# <source>/<benchmark_name>/<files...>
cherrybin build local.db /path/to/source_tree

# See what's in it
cherrybin list --db local.db

# Publish it to the shared drive
cherrybin publish local.db /mnt/shared/bench_archives --version v1

# Elsewhere (or later): check out just one benchmark
cherrybin checkout my_benchmark ./workdir/my_benchmark ./blob_cache \
    --shared_dir /mnt/shared/bench_archives

# Update one benchmark's files and republish
cherrybin build local.db /path/to/source_tree --benchmark my_benchmark
cherrybin publish local.db /mnt/shared/bench_archives --version v2

# Drop a retired benchmark and reclaim its now-unreferenced blobs
cherrybin remove local.db old_benchmark
cherrybin gc local.db
```

`cherrybin checkout` and `cherrybin list` both accept either `--db
<path>` (a direct path to an archive file) or `--shared_dir <dir>` (a
directory containing `current.txt`, always resolving to the latest
published version).

## Development

```bash
make install   # editable install with dev dependencies
make test      # pytest
make lint      # ruff
```

## License

BSD-3-Clause
