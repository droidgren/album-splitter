from albumsplit import __version__, version
from albumsplit.__main__ import main


def _fresh(monkeypatch, **env):
    for k in ("ALBUMSPLIT_COMMIT", "ALBUMSPLIT_COMMIT_DATE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    version.build_info.cache_clear()


def test_docker_env_wins(monkeypatch):
    _fresh(monkeypatch, ALBUMSPLIT_COMMIT="a90198b1234567890", ALBUMSPLIT_COMMIT_DATE="2026-11-24")
    assert version.version_string() == f"{__version__}+a90198b (2026-11-24)"
    version.build_info.cache_clear()


def test_no_git_is_plain_version(monkeypatch, tmp_path):
    _fresh(monkeypatch)
    monkeypatch.setattr(version, "REPO", tmp_path)
    assert version.version_string() == __version__
    version.build_info.cache_clear()


def test_cli_flag(monkeypatch, capsys):
    _fresh(monkeypatch, ALBUMSPLIT_COMMIT="abcdef0")
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    assert capsys.readouterr().out.strip() == f"albumsplit {__version__}+abcdef0"
    version.build_info.cache_clear()
