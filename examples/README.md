# Example inputs

These complete customization artifacts can be passed to the current CLI:

- [Context-only summary](in/summary.instructions.md): a small response-only example.
- [Debug agent](in/debug.agent.md)
- [Detailed agent](in/ultimate-transparent-thinking-beast-mode.agent.md)
- [Caveman code-review skill](in/caveman-review/SKILL.md): an upstream MIT-licensed
	response-formatting sample; see its [provenance and license](in/caveman-review/README.md).

They are input samples, not verified executable agents or evidence of optimization
quality. Zen evaluates their instruction bodies in tool-free sessions; it does not
run the workflows or tools described inside them.

## Run an example

From the repository root, with the Copilot CLI signed in:

```powershell
uv run zen detect .\examples\in\debug.agent.md
uv run zen --budget 32 optimize .\examples\in\debug.agent.md --output-dir .\examples\out --quick
```

The default semantic engine rewrites the complete instruction body, saves a draft,
then checks instruction meaning and compares original/candidate answers on a small
generated case set. `--quick` uses fewer cases. It is an illustrative check, not proof
of equivalence for all future tasks.

The command saves a usable draft before evaluation and reports whether verification
completed successfully, final adjudication confirmed disputed findings, or manual
review is still needed. CONFIRMED is a model-review outcome, not human approval.
A later evaluator error does not
discard the draft. Initial generation or filesystem failure can still prevent a draft.
Generated results are local and ignored by Git. Review the report and candidate;
never assume that a smaller body preserves all future behavior.

Semantic mode has no hard line cap by default.
Use `--aggressive 80` to cap the whole mutable body at 80 lines, or
`--aggressive 50%` for half its original line count, rounded up. Caps do not relax
semantic verification. GEPA search remains available with `--engine gepa`, with its
own budget and acceptance-only output policy. See the
[runtime guide](../docs/documentation.md) for all options.