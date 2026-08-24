"""`cherrybin list` - show benchmarks stored in an archive."""

from __future__ import annotations

from dataclasses import dataclass

from argklass.arguments import add_arguments
from argklass.command import Command, newparser

from cherrybin.core import connect_writable, list_benchmarks, resolve_current


@dataclass
class Arguments:
    db: str = ""
    shared_dir: str = ""


class List(Command):
    """List benchmarks present in an archive, with file counts and sizes."""

    name: str = "list"

    @staticmethod
    def arguments(subparsers):
        parser = newparser(subparsers, List)
        add_arguments(parser, Arguments)

    def execute(self, args: Arguments):
        if args.db:
            db_path = args.db
        elif args.shared_dir:
            db_path = resolve_current(args.shared_dir)
        else:
            print("error: pass either --db or --shared-dir")
            return 1

        con = connect_writable(db_path)
        try:
            for stats in list_benchmarks(con):
                print(f"{stats.name:30s} {stats.file_count:6d} files  {stats.total_bytes / 1e6:9.1f} MB")
        finally:
            con.close()
        return 0


COMMANDS = [List]
