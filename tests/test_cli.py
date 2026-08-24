"""CLI wrappers around cherrybin.core."""

from cherrybin.cli.update import Arguments, Update, parse_root_spec


def test_parse_root_spec():
    assert parse_root_spec("/tmp/data") == ("/tmp/data", "")
    assert parse_root_spec("/tmp/data:data") == ("/tmp/data", "data")
    assert parse_root_spec("/tmp/cache:cache") == ("/tmp/cache", "cache")


def test_update_cli_creates_db(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "file.bin").write_text("hello")
    db = tmp_path / "archive.db"

    args = Arguments(
        db=str(db),
        benchmark="bench",
        root=[f"{tree}:data"],
    )
    assert Update().execute(args) == 0
    assert db.is_file()

    from cherrybin.core import list_files

    assert list_files(str(db), "bench")[0][0] == "data/file.bin"
