"""`cherrybin checkout` - materialize a benchmark's files locally."""

from __future__ import annotations

from dataclasses import dataclass

from argklass.arguments import add_arguments
from argklass.command import Command, newparser

from cherrybin.core import DEFAULT_IO_CHUNK, checkout, resolve_current


@dataclass
class Arguments:
    benchmark: str
    dest: str
    cache: str
    db: str = ""
    shared_dir: str = ""
    io_chunk: int = DEFAULT_IO_CHUNK


class Checkout(Command):
    """Materialize one benchmark's files from the archive into --dest.

    Pass either --db (a direct path to an archive file) or --shared-dir
    (a directory containing current.txt, to always use the latest
    published version). Unchanged files already present in --cache are
    hardlinked rather than re-read from the archive.
    """

    name: str = "checkout"

    @staticmethod
    def arguments(subparsers):
        parser = newparser(subparsers, Checkout)
        add_arguments(parser, Arguments)

    def execute(self, args: Arguments):
        if args.db:
            db_path = args.db
        elif args.shared_dir:
            db_path = resolve_current(args.shared_dir)
        else:
            print("error: pass either --db or --shared-dir")
            return 1

        try:
            result = checkout(
                db_path, args.benchmark, args.dest, args.cache, io_chunk=args.io_chunk
            )
        except (FileNotFoundError, KeyError) as e:
            print(f"error: {e}")
            return 1

        print(
            f"[{result.benchmark}] {result.file_count} files -> {result.dest} "
            f"({result.pulled_from_archive} pulled from archive, "
            f"{result.already_cached} already cached)"
        )
        return 0


COMMANDS = [Checkout]
