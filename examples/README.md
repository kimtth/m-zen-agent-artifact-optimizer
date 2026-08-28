# Real input and output

This directory contains complete customization artifacts and the output of a Zen quick run.

## Accepted example

- Input: `in/ultimate-transparent-thinking-beast-mode.agent.md`
- Candidate: `out/ultimate-transparent-thinking-beast-mode.agent.optimized.md`
- Report: `out/ultimate-transparent-thinking-beast-mode.agent.optimize.report.md`

The recorded run accepted the candidate. It reduced artifact tokens from 6,830 to 941 and
median output tokens from 64.5 to 59, for an 85.5% communication-token reduction. The
report records the run's behavior, critical-failure, understanding, and model-call measures.

Only the natural-language body is eligible for optimization. The output preserves the input
frontmatter, UTF-8 BOM when present, and line-ending convention.

## Reproduce

Run from the repository root:

```powershell
uv run zen --target-model gpt-5.6-terra --strong-model gpt-5.6-sol --generator-model gpt-5.6-luna --budget 100 --max-metric-calls 6 --seed 0 --cache-dir examples\out\.zen-cache optimize examples\in\ultimate-transparent-thinking-beast-mode.agent.md --output-dir examples\out --quick
```

The signed-in GitHub Copilot account supplies the models. Model nondeterminism can produce
a different candidate or a REJECT decision even with the same seed. This command uses the
quick profile, which is illustrative rather than full-profile acceptance evidence.

## Aggressive skill mode

Append `--aggressive 80` to an `optimize` command to create a minimal but effective AI agent
skill definition capped at 80 lines. Use `--aggressive 50%` to cap the body at half its
original line count, rounded up. `--aggressive` without a value keeps the 100-line default;
every option still requires the normal quality gate to pass.