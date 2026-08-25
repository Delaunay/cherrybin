"""`cherrybin update` - create or incrementally update a shared archive file."""

from __future__ import annotations

from dataclasses import dataclass

from argklass.arguments import add_arguments, argument
from argklass.command import Command, newparser

from cherrybin.core import DEFAULT_IO_CHUNK, update_file


def parse_root_spec(spec: str) -> tuple[str, str]:
    """Parse ``DIR`` or ``DIR:PREFIX`` into ``(abs-or-given dir, prefix)``."""
    if ":" in spec:
        root, prefix = spec.rsplit(":", 1)
        if prefix and "/" not in prefix:
            return root, prefix
    return spec, ""


@dataclass
class Arguments:
    db: str
    benchmark: str
    root: list[str] = argument(default=[], action="append")
    lock_timeout: float = 600.0
    io_chunk: int = DEFAULT_IO_CHUNK


class Update(Command):
    """Create or incrementally update --db from one or more --root DIR[:PREFIX] trees.

    Adds new files, drops files that vanished for this benchmark, and
    leaves other benchmarks alone. Writes --db via an atomic replace
    (never mutates the live file in place).
    """

    name: str = "update"

    @staticmethod
    def arguments(subparsers):
        parser = newparser(subparsers, Update)
        add_arguments(parser, Arguments)

    def execute(self, args: Arguments):
        roots = args.root or []
        if not roots:
            print("error: pass at least one --root DIR[:PREFIX]")
            return 1

        parsed = [parse_root_spec(spec) for spec in roots]
        stats = update_file(
            args.db,
            args.benchmark,
            parsed,
            lock_timeout=args.lock_timeout,
            io_chunk=args.io_chunk,
        )

        print(
            f"[{stats.name}] {stats.file_count} files "
            f"(+{stats.added} -{stats.removed} ={stats.unchanged}) "
            f"new_blobs={stats.new_bytes / 1e6:.1f} MB "
            f"{'updated' if stats.changed else 'unchanged'} {args.db}"
        )
        return 0


COMMANDS = [Update]
