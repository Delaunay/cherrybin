"""`cherrybin gc` and `cherrybin remove` - archive maintenance."""

from __future__ import annotations

from dataclasses import dataclass

from argklass.arguments import add_arguments
from argklass.command import Command, newparser

from cherrybin.core import connect_writable, gc_unreferenced_blobs, remove_benchmark


@dataclass
class RemoveArguments:
    db: str
    benchmark: str


class Remove(Command):
    """Remove a benchmark's file listing from the (local) archive.

    Blobs are kept if another benchmark still references them; run
    `cherrybin gc` afterwards to actually reclaim unreferenced space.
    """

    name: str = "remove"

    @staticmethod
    def arguments(subparsers):
        parser = newparser(subparsers, Remove)
        add_arguments(parser, RemoveArguments)

    def execute(self, args: RemoveArguments):
        con = connect_writable(args.db)
        try:
            n = remove_benchmark(con, args.benchmark)
        finally:
            con.close()
        print(f"[{args.benchmark}] removed {n} file entries (blobs kept if referenced elsewhere)")
        return 0


@dataclass
class GcArguments:
    db: str


class Gc(Command):
    """Reclaim space from blobs no longer referenced by any benchmark."""

    name: str = "gc"

    @staticmethod
    def arguments(subparsers):
        parser = newparser(subparsers, Gc)
        add_arguments(parser, GcArguments)

    def execute(self, args: GcArguments):
        con = connect_writable(args.db)
        try:
            n = gc_unreferenced_blobs(con)
        finally:
            con.close()
        print(f"removed {n} unreferenced blobs")
        return 0


COMMANDS = [Remove, Gc]
