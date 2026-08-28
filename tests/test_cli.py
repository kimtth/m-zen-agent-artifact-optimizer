from io import StringIO

from zen import __version__
from zen.cli import _aggressive_limit, main, parser
from zen.runtime.progress import ProgressBar


def test_version_is_new_rebuild_version(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert capsys.readouterr().out.strip() == f"zen {__version__}"


def test_selfcheck_passes() -> None:
    assert main(["selfcheck"]) == 0


def test_optimize_accepts_output_directory() -> None:
    args = parser().parse_args(
        ["optimize", "AGENTS.md", "--output-dir", "out", "--quick", "--aggressive", "50%"]
    )
    assert str(args.output_dir) == "out"
    assert args.quick is True
    assert args.aggressive.percent == 50


def test_aggressive_limit_accepts_lines_and_percentages() -> None:
    assert _aggressive_limit("80").lines == 80
    assert _aggressive_limit("50%").percent == 50


def test_progress_bar_renders_completion() -> None:
    stream = StringIO()
    progress = ProgressBar(stream, width=10)

    progress.update(0, "Starting")
    progress.update(50, "Working")
    progress.update(100, "Finished")
    progress.close()

    output = stream.getvalue()
    assert "[#####-----]  50% Working" in output
    assert "[##########] 100% Finished" in output
    assert output.endswith("\n")
    assert "\r" not in output
