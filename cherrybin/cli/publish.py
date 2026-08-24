"""`cherrybin publish` - push a local archive to the shared drive."""

from __future__ import annotations

import time
from dataclasses import dataclass

from argklass.arguments import add_arguments
from argklass.command import Command, newparser

from cherrybin.core import publish, publish_lock


@dataclass
class Arguments:
    local: str
    shared_dir: str
    version: str = ""
    lock_timeout: float = 600.0


class Publish(Command):
    """Publish a locally-built archive as a new immutable version.

    Copies --local to --shared-dir under a versioned name, then
    atomically flips the 'current' pointer. Readers only ever see a
    fully-written file, so this is safe to run while other processes
    are checking out benchmarks from the same shared directory.
    """

    name: str = "publish"

    @staticmethod
    def arguments(subparsers):
        parser = newparser(subparsers, Publish)
        add_arguments(parser, Arguments)

    def execute(self, args: Arguments):
        version = args.version or time.strftime("%Y%m%d_%H%M%S")
        try:
            with publish_lock(args.shared_dir, timeout=args.lock_timeout):
                path = publish(args.local, args.shared_dir, version)
        except TimeoutError as e:
            print(f"error: {e}")
            return 1
        except FileExistsError as e:
            print(f"error: {e}")
            return 1

        print(f"published {path}")
        print(f"updated pointer: {args.shared_dir}/current.txt -> archive_{version}.db")
        return 0


COMMANDS = [Publish]
