"""`cherrybin build` - create or update a local archive from a source tree."""

from __future__ import annotations

import os
from dataclasses import dataclass

from argklass.arguments import add_arguments
from argklass.command import Command, newparser

from cherrybin.core import DEFAULT_IO_CHUNK, add_benchmark, connect_writable


@dataclass
class Arguments:
    db: str
    source: str
    benchmark: str = ""
    io_chunk: int = DEFAULT_IO_CHUNK


class Build(Command):
    """Build or update the local archive from <source>/<benchmark>/** trees.

    Always run this against a LOCAL path for --db (fast disk, real file
    locking). Use `cherrybin publish` afterwards to push the result to
    the shared drive.
    """

    name: str = "build"

    @staticmethod
    def arguments(subparsers):
        parser = newparser(subparsers, Build)
        add_arguments(parser, Arguments)

    def execute(self, args: Arguments):
        con = connect_writable(args.db)
        try:
            if args.benchmark:
                names = [args.benchmark]
            else:
                names = sorted(
                    n for n in os.listdir(args.source)
                    if os.path.isdir(os.path.join(args.source, n))
                )

            for name in names:
                stats = add_benchmark(con, args.source, name, io_chunk=args.io_chunk)
                print(
                    f"[{stats.name}] indexed {stats.file_count} files, "
                    f"wrote {stats.total_bytes / 1e6:.1f} MB of new blob data"
                )
        finally:
            con.close()
        return 0


COMMANDS = [Build]
