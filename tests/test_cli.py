from io import StringIO

import pytest

from zen import __version__
from zen.cli import _aggressive_limit, main, parser
from zen.domain.core import OptimizeConfig
from zen.runtime.progress import ProgressBar


def test_version_is_new_rebuild_version(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert capsys.readouterr().out.strip() == f"zen {__version__}"


def test_selfcheck_passes() -> None:
    assert main(["selfcheck"]) == 0


def test_requested_model_defaults_match_cli_and_config(monkeypatch) -> None:
    for role in ("TARGET", "STRONG", "GENERATOR"):
        monkeypatch.delenv(f"ZEN_{role}_MODEL", raising=False)
    args = parser().parse_args(["optimize", "AGENTS.md"])
    config = OptimizeConfig()
    assert args.target_model == config.target_model == "gpt-5.6-terra"
    assert args.strong_model == config.strong_model == "gpt-5.6-sol"
    assert args.generator_model == config.generator_model == "gpt-5.6-luna"


def test_model_overrides_remain_explicit(monkeypatch) -> None:
    for role in ("TARGET", "STRONG", "GENERATOR"):
        monkeypatch.setenv(f"ZEN_{role}_MODEL", f"env-{role.lower()}")
    args = parser().parse_args(["optimize", "AGENTS.md"])
    assert (args.target_model, args.strong_model, args.generator_model) == (
        "env-target", "env-strong", "env-generator",
    )
    args = parser().parse_args([
        "--target-model", "chosen-target", "--strong-model", "chosen-strong",
        "--generator-model", "chosen-weak", "optimize", "AGENTS.md",
    ])
    assert (args.target_model, args.strong_model, args.generator_model) == (
        "chosen-target", "chosen-strong", "chosen-weak",
    )


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


def test_semantic_defaults_and_optional_aggressive_cap() -> None:
    args = parser().parse_args(["optimize", "AGENTS.md"])
    config = OptimizeConfig()
    assert args.engine == config.engine == "semantic"
    assert args.budget == config.total_call_budget == 32
    assert args.aggressive is config.aggressive_limit is None
    assert config.max_body_lines("First.\nSecond.\n") is None
    assert not config.aggressive
    assert config.max_metric_calls is None
    assert parser().parse_args([
        "optimize", "AGENTS.md", "--aggressive",
    ]).aggressive.percent == 50


def test_gepa_aggressive_default_can_be_disabled() -> None:
    args = parser().parse_args(["optimize", "AGENTS.md", "--engine", "gepa"])
    config = OptimizeConfig(engine="gepa")
    assert args.budget == config.total_call_budget == 600
    assert config.max_metric_calls == 120
    assert args.aggressive == config.aggressive_limit
    assert args.aggressive.lines is None
    assert args.aggressive.percent == 50
    assert config.aggressive
    assert parser().parse_args(["optimize", "AGENTS.md", "--aggressive"]).aggressive == args.aggressive
    for original_lines, expected in ((1, 1), (101, 51), (400, 200)):
        body = "\n".join("Instruction." for _ in range(original_lines))
        assert config.max_body_lines(body) == expected
    assert parser().parse_args([
        "optimize", "AGENTS.md", "--aggressive", "30%",
    ]).aggressive.percent == 30
    args = parser().parse_args(["optimize", "AGENTS.md", "--engine", "gepa", "--no-aggressive"])
    assert args.aggressive is None
    assert not OptimizeConfig(engine="gepa", aggressive_limit=None).aggressive


def test_optimize_accepts_diagnostic_and_scope_flags() -> None:
    args = parser().parse_args([
        "optimize", "AGENTS.md", "--compare-concision", "--focus", "communication",
    ])
    assert args.compare_concision
    assert args.focus == "communication"


def test_percentage_cap_selection_accepts_gepa_program_candidates() -> None:
    from types import SimpleNamespace

    from zen.optimization.service import _within_limit
    from zen.runtime.harness import ArtifactProgram, CandidatePolicy

    source = "Explain.\nRepeat.\nRepeat.\nRepeat.\n"
    shorter = "Explain.\n"
    config = OptimizeConfig(engine="gepa")
    programs = [ArtifactProgram(body, CandidatePolicy(source)) for body in (source, shorter)]
    optimized = SimpleNamespace(detailed_results=SimpleNamespace(
        candidates=programs, val_aggregate_scores=[0.99, 0.95],
    ))
    assert _within_limit(optimized, source, config.max_body_lines(source)) == shorter
    assert _within_limit(optimized, source, 0) == source


@pytest.mark.parametrize("arguments,engine,budget", [
    (["--budget", "32", "optimize", "AGENTS.md", "--engine", "semantic"], "semantic", 32),
    (["--engine", "gepa", "optimize", "AGENTS.md"], "gepa", 600),
    (["--engine", "gepa", "optimize", "AGENTS.md", "--engine", "semantic"], "semantic", 32),
    (["optimize", "AGENTS.md", "--engine", "gepa", "--budget", "700"], "gepa", 700),
])
def test_engine_and_budget_routing(arguments, engine, budget, monkeypatch) -> None:
    monkeypatch.delenv("ZEN_BUDGET", raising=False)
    args = parser().parse_args(arguments)
    assert args.engine == engine
    assert args.budget == budget


@pytest.mark.parametrize("flag", [
    ["--max-metric-calls", "120"],
    ["--compare-concision"],
])
def test_semantic_rejects_gepa_only_options(flag, monkeypatch, capsys) -> None:
    def unexpected(*args):
        pytest.fail("incompatible options must fail before optimization")

    monkeypatch.setattr("zen.cli.optimize", unexpected)
    assert main(["optimize", "AGENTS.md", *flag]) == 1
    assert "require --engine gepa" in capsys.readouterr().err


def test_semantic_rejects_metric_budget_environment(monkeypatch, capsys) -> None:
    monkeypatch.setenv("ZEN_MAX_METRIC_CALLS", "120")
    assert main(["optimize", "AGENTS.md"]) == 1
    assert "--max-metric-calls" in capsys.readouterr().err


@pytest.mark.parametrize("decision,status", [
    ("VERIFIED", 0), ("CONFIRMED", 0), ("REVIEW_REQUIRED", 1), ("ERROR", 1),
    ("ACCEPT", 0), ("REJECT", 0), ("INCONCLUSIVE", 1),
])
def test_cli_passes_output_directory_before_optimization(
    decision, status, monkeypatch, tmp_path,
) -> None:
    from types import SimpleNamespace

    monkeypatch.delenv("ZEN_MAX_METRIC_CALLS", raising=False)
    output_dir = tmp_path / "out"

    def fake_optimize(path, config, cache_root, progress):
        assert config.output_dir == output_dir
        assert config.quick
        assert config.engine == "semantic"
        assert config.total_call_budget == 32
        assert config.aggressive_limit is None
        return SimpleNamespace(decision=decision, message="Result", run_directory="run")

    monkeypatch.setattr("zen.cli.optimize", fake_optimize)
    monkeypatch.setattr("zen.cli.write_outputs", lambda result, output: (None, output / "report.md"))
    assert main([
        "--budget", "32", "optimize", "AGENTS.md", "--engine", "semantic",
        "--output-dir", str(output_dir), "--quick",
    ]) == status


@pytest.mark.parametrize("engine", ["semantic", "gepa"])
def test_factory_routes_engines_with_shared_budget(engine, monkeypatch, tmp_path) -> None:
    import json
    import sys
    from dataclasses import asdict
    from types import ModuleType

    from zen.optimization import service

    config = OptimizeConfig(engine=engine, quick=True, output_dir=tmp_path / "out")
    path = tmp_path / "AGENTS.md"
    cache = tmp_path / "cache"
    models = []
    result = object()

    def model(model_id, budget, **kwargs):
        value = (model_id, budget, kwargs)
        models.append(value)
        return value

    class RoutedOptimizer:
        def __init__(self, received, task, strong, generator, budget, cache_root, progress):
            assert received.engine == engine
            assert received.quick
            if engine == "semantic":
                assert received.output_dir == config.output_dir
            else:
                assert received.output_dir is None
                json.dumps(asdict(received))
            assert [task, strong, generator] == models
            assert all(item[1] is budget for item in models)
            assert cache_root == cache

        def run(self, artifact_path):
            assert artifact_path == path.resolve()
            return result

    def unexpected(*args, **kwargs):
        pytest.fail("wrong optimizer selected")

    semantic = ModuleType("zen.optimization.semantic")
    semantic.SemanticOptimizer = RoutedOptimizer if engine == "semantic" else unexpected
    monkeypatch.setitem(sys.modules, "zen.optimization.semantic", semantic)
    monkeypatch.setattr(service, "Optimizer", RoutedOptimizer if engine == "gepa" else unexpected)
    monkeypatch.setattr(service, "CopilotModel", model)
    assert service.optimize(path, config, cache) is result
    assert len(models) == 3


@pytest.mark.parametrize("kwargs", [
    {"max_metric_calls": 120}, {"compare_concision": True},
])
def test_factory_rejects_gepa_settings_for_semantic(kwargs, tmp_path) -> None:
    from zen.optimization.service import optimize

    with pytest.raises(ValueError, match="require engine='gepa'"):
        optimize(tmp_path / "AGENTS.md", OptimizeConfig(**kwargs))


def test_write_outputs_routes_to_semantic_writer(monkeypatch, tmp_path) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    from zen.optimization.service import write_outputs

    result = SimpleNamespace(details={"engine": "semantic"})
    expected = (tmp_path / "candidate.md", tmp_path / "report.md")
    semantic = ModuleType("zen.optimization.semantic")

    def write_semantic_outputs(received, output_dir=None):
        assert received is result
        assert output_dir == tmp_path
        return expected

    semantic.write_semantic_outputs = write_semantic_outputs
    monkeypatch.setitem(sys.modules, "zen.optimization.semantic", semantic)
    assert write_outputs(result, tmp_path) == expected


def test_semantic_focus_reaches_optimizer(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    monkeypatch.delenv("ZEN_MAX_METRIC_CALLS", raising=False)

    def optimize(path, config, cache, progress):
        assert config.focus == "task"
        assert config.engine == "semantic"
        return SimpleNamespace(decision="VERIFIED", message="OK", run_directory="run")

    monkeypatch.setattr("zen.cli.optimize", optimize)
    monkeypatch.setattr("zen.cli.write_outputs", lambda result, output: (None, tmp_path / "report.md"))
    assert main(["optimize", "AGENTS.md", "--focus", "task"]) == 0


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


@pytest.mark.parametrize("answer,expected,status", [
    ("yes", "ACCEPTED", 0), ("no", "DECLINED", 1), ("", "DECLINED", 1),
])
def test_cli_asks_before_user_confirmation(answer, expected, status, monkeypatch, tmp_path, capsys):
    from test_semantic import _json, _make

    monkeypatch.delenv("ZEN_MAX_METRIC_CALLS", raising=False)
    harness = _make(tmp_path, budget=1)
    result = harness.run()
    monkeypatch.setattr("zen.cli.optimize", lambda *args: result)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    prompts = []

    def respond(prompt):
        prompts.append(prompt)
        assert "Report:" in capsys.readouterr().out
        return answer

    monkeypatch.setattr("builtins.input", respond)
    assert main(["optimize", str(harness.source)]) == status
    assert len(prompts) == 1
    assert "[y/N]" in prompts[0]
    assert result.details["user_acceptance"]["status"] == expected
    summary = _json(harness.output / "AGENTS.optimize.manifest.json")
    assert summary["decision"] == ("CONFIRMED" if answer == "yes" else "REVIEW_REQUIRED")
    if answer == "yes":
        assert "CONFIRMED (by user)" in capsys.readouterr().out
        assert summary["confirmation_origin"] == "user"


@pytest.mark.parametrize("explicit", [True, False])
def test_noninteractive_cli_requires_explicit_consent(explicit, monkeypatch, tmp_path):
    from test_semantic import _make

    monkeypatch.delenv("ZEN_MAX_METRIC_CALLS", raising=False)
    harness = _make(tmp_path, budget=1)
    result = harness.run()
    monkeypatch.setattr("zen.cli.optimize", lambda *args: result)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("must not read redirected input"))
    flags = ["--accept-draft"] if explicit else []
    assert main(["optimize", str(harness.source), *flags]) == (0 if explicit else 1)
    assert result.decision == ("CONFIRMED" if explicit else "REVIEW_REQUIRED")


def test_accept_draft_does_not_bypass_gepa(monkeypatch, capsys):
    monkeypatch.setattr("zen.cli.optimize", lambda *args: pytest.fail("must fail before model calls"))
    assert main(["optimize", "AGENTS.md", "--engine", "gepa", "--accept-draft"]) == 1
    assert "requires --engine semantic" in capsys.readouterr().err


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_interrupted_acceptance_never_confirms(exception, monkeypatch, tmp_path):
    from test_semantic import _make

    monkeypatch.delenv("ZEN_MAX_METRIC_CALLS", raising=False)
    harness = _make(tmp_path, budget=1)
    result = harness.run()
    monkeypatch.setattr("zen.cli.optimize", lambda *args: result)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def interrupted(prompt):
        raise exception

    monkeypatch.setattr("builtins.input", interrupted)
    assert main(["optimize", str(harness.source)]) == (130 if exception is KeyboardInterrupt else 1)
    assert result.decision == "REVIEW_REQUIRED"
    assert result.details.get("user_acceptance", {}).get("status") != "ACCEPTED"
