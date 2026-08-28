# Zen

<img src="docs/zen-logo.svg" alt="Zen logo" width="112" align="left" />

<p><strong>Less input. Less output. More clarity.</strong></p>
<p>Zen reduces the instructions an AI agent reads and the output a person must read. It optimizes one GitHub Copilot customization artifact without a hand-written dataset.</p>

<br clear="left" />

Accepts a candidate only when it:

1. introduces no critical behavior failure;
2. preserves behavior pass count;
3. preserves reader understanding;
4. does not increase artifact, output, or understanding tokens; and
5. reduces combined artifact and median output tokens by at least 3%.

The source artifact is never modified.

## What is different

- **No hard-coded behavioral rule set.** A model derives atomic, source-grounded obligations from each artifact. Evaluators consume that generated contract.
- **Multilingual.** Contracts, synthetic inquiries, criteria, and reader questions stay in the artifact's language. Unicode writing-system checks reject accidental translation.
- **Dataset-free.** Zen generates 80 cases, validates and deduplicates them, retains 50, and splits by scenario family into 30 train, 10 validation, and 10 sealed holdout cases.
- **Quality before size.** GEPA receives behavior, understanding, and efficiency feedback. Shorter incorrect candidates cannot outrank correct candidates.

## Install and verify

```powershell
uv sync
python -m copilot download-runtime
uv run pytest
uv run zen selfcheck
```

The Copilot SDK uses the signed-in Copilot CLI account.

## Optimize

```powershell
uv run zen --target-model gpt-5-mini --strong-model gpt-5 --budget 600 optimize .github\copilot-instructions.md
```

Useful commands:

```powershell
uv run zen detect .github\copilot-instructions.md
uv run zen selfcheck
uv run zen --help
```

A completed accepted run writes:

- `NAME.optimized.<suffix>` — the accepted candidate;
- `NAME.optimize.report.md` — the concise decision and measurements;
- `.zen-cache/runs/<run-id>/` — generated contract, frozen dataset, holdout seal, and detailed results.

A rejected run writes the report and retains the source. Zen does not automatically apply any candidate.
During processing, the CLI displays the current phase and percentage. Pass
`--output-dir DIRECTORY` after `PATH` to place the candidate and report somewhere other
than beside the source artifact. Add `--quick` for a low-cost illustrative run; use the
default full profile for acceptance evidence. Add `--aggressive 80` to cap the mutable
body at 80 lines, or `--aggressive 50%` to cap it at half of its original line count.
Without a value, `--aggressive` defaults to 100 lines. The usual quality gate still
applies, so Zen rejects a shorter candidate that loses behavior.

## Supported artifacts

- `AGENTS.md`
- `copilot-instructions.md`
- `*.instructions.md`
- `*.prompt.md`
- `*.agent.md`
- `SKILL.md`

See the consolidated [documentation](docs/documentation.md).

## Real-world example

The `examples` directory contains complete input artifacts and the accepted output from an
actual quick Zen run. See [examples/README.md](examples/README.md) for the input, command,
and recorded measurements.
