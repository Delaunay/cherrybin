# cherrybin

> SQLite-backed, content-addressed archive for selective checkout of large shared benchmark datasets

## Project structure

```
cherrybin/                 ← Project root (git repo)
├── pyproject.toml         ← Package config (setuptools)
├── Makefile                ← Dev commands
├── tests/                  ← pytest test suite
│   ├── conftest.py         ← source_tree fixture (two benchmarks, one shared file)
│   └── test_core.py        ← build/dedup/checkout/publish/gc round-trip tests
└── cherrybin/               ← Python package
    ├── __init__.py          ← Namespace package (extend_path)
    ├── core.py              ← All archive logic (schema, build, publish, checkout, gc)
    ├── cli/                 ← CLI entry point (argklass)
    │   ├── __init__.py      ← Command discovery + main()
    │   ├── build.py         ← `cherrybin build`
    │   ├── publish.py       ← `cherrybin publish`
    │   ├── checkout.py      ← `cherrybin checkout`
    │   ├── list.py           ← `cherrybin list`
    │   └── gc.py             ← `cherrybin remove` / `cherrybin gc`
    └── plugins/              ← Namespace package for extensions (empty by default)
        └── __init__.py       ← extend_path hook
```

## Setup

Prerequisites: Python >= 3.11

```bash
python -m venv .venv
source .venv/bin/activate
make install   # editable install with dev dependencies
```

## Key design points (see also README.md)

- `cherrybin/core.py` is the single source of truth for all archive
  behavior; CLI commands in `cherrybin/cli/` are thin wrappers that
  parse arguments and call into `core.py`. Anything reusable (e.g. a
  future Python API, a web dashboard) should import `core` directly
  rather than shelling out to the CLI.
- Archives are single SQLite files with two tables: `blobs` (hash ->
  content, deduped) and `benchmark_files` (benchmark, relpath -> hash).
  See `core.SCHEMA`.
- **Writers never touch the shared copy directly.** `build()` /
  `add_benchmark()` always run against a local path. `publish()` copies
  the finished file to the shared drive under a new versioned name and
  atomically flips `current.txt` via `os.replace`.
- **Readers never need a lock.** Because publish is atomic and
  versioned, a reader either sees the old `current.txt` or the new one,
  and whichever archive file it opens is always fully written.
  `checkout()` opens archives with `mode=ro&immutable=1`.
- The only lock in the codebase (`core.publish_lock`) guards against
  two *writers* publishing at the same time. It uses `O_CREAT|O_EXCL`
  rather than `flock`/`fcntl`, since real file locks are unreliable
  over NFS/SMB-style network filesystems — this project assumes the
  shared drive is exactly that kind of filesystem.
- `cherrybin.cli` command discovery expects each `cli/*.py` module to
  export a module-level `COMMANDS` list of `argklass.command.Command`
  subclasses (see any existing file in `cli/` for the pattern).

## Testing

```bash
make test    # pytest
make lint    # ruff check
make format  # ruff fix + format
```

`tests/conftest.py` provides a `source_tree` fixture: two benchmarks
(`bench_a`, `bench_b`) sharing one file (`common.bin`) plus one unique
file each, which is enough to exercise dedup, checkout, and
cross-benchmark hardlinking without needing real benchmark data.
