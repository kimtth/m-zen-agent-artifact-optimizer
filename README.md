# Zen

<img src="docs/zen-logo.svg" alt="Zen logo" width="112" align="left" />

<p><strong>Less input. Less output. More clarity.</strong></p>
<p>Zen rewrites GitHub Copilot instructions more compactly, saves a reviewable draft, and checks for meaningful differences. No user-supplied labeled dataset is required.</p>

<br clear="left" />

## What Zen does

Zen optimizes **one Markdown instruction body** using tool-free Copilot sessions.

- **Rewrite the meaning, not just the lines.** The default semantic engine reads
  the whole body and consolidates repetition while preserving actions, constraints,
  conditions, exceptions, priorities, language, and required public output formats.
- **Save before judging.** A usable draft is written before evaluation. Later
  evaluation errors or budget exhaustion do not discard it.
- **Compare meaning.** Direct instruction review and small task-based comparisons
  look for important omissions, contradictions, and behavioral regressions, not
  identical wording or mandatory quotations.
- **Bound the work.** A small call budget and limited correction rounds replace
  open-ended deletion/search as the default.
- **Keep the source unchanged.** Metadata and source bytes are protected; generated
  drafts are separate files, never automatically applied.

Zen supports AGENTS.md, copilot-instructions.md, SKILL.md, and files ending in
`.instructions.md`, `.prompt.md`, or `.agent.md`. YAML frontmatter, UTF-8 BOM, and
line-ending convention are preserved.

## Install and verify

Requires Python 3.13 or later, uv, and a signed-in Copilot CLI account.

```powershell
uv sync
uv run python -m copilot download-runtime
uv run zen selfcheck
```

## Use

```powershell
uv run zen detect .\examples\in\debug.agent.md
uv run zen --budget 32 optimize .\examples\in\debug.agent.md --output-dir .\examples\out --quick
```

Global options such as `--budget` precede `optimize`. Default model roles are:

- Target: `gpt-5.6-terra` (`--target-model`).
- Strong: `gpt-5.6-sol` for rewriting and semantic judgments (`--strong-model`).
- Generator: `gpt-5.6-luna` for synthetic cases (`--generator-model`).

| Optimize option | Effect |
| --- | --- |
| `--engine semantic` | Whole-body semantic rewriting and bounded comparison; the default. |
| `--engine gepa` | Opt into GEPA search and its separate evaluation policy. |
| `--quick` | Smaller case set for an illustrative comparison. |
| `--output-dir DIRECTORY` | Write the draft and report in this directory. |
| `--aggressive 30%` | Request a body line cap. Semantic rewriting has no hard line cap by default. |
| `--no-aggressive` | Disable the body line cap. |
| `--focus task` or `--focus communication` | Edit only an explicitly marked section; outside text stays frozen. Default: `all`. |
| `--compare-concision` | GEPA-only diagnostic control; never a selected artifact. |

Focused edits require an exact `<!-- zen:task --> ... <!-- /zen:task -->` or
`<!-- zen:communication --> ... <!-- /zen:communication -->` pair. There is no
automatic semantic split.

The budget caps application model calls, not dollars or provider tokens. If the
remaining budget cannot complete the checks, the draft remains available with a
review-required status. See the [CLI and runtime guide](docs/documentation.md).

## Decisions and outputs

The semantic engine separates **having a draft** from **having sufficient verification**.
Harmless differences in wording, order, or length are not failures. Important missing
instructions, incorrect added requirements, or materially worse answers require review.
An original answer's existing defect is not automatically a candidate regression.

The report identifies instruction-level differences, validation and holdout results,
local token counts, errors, and output availability without filesystem paths.
Locations remain in the CLI and machine-readable manifest. Successful
verification is limited to these checks, not universal semantic equivalence.
Incomplete checks never become a successful score.

If complete evaluation flags a draft for review, a **final adjudication** compares
the frozen source, draft, cases, and both answers to distinguish new regressions
from shared baseline defects or evaluator mistakes. It may return **CONFIRMED**
after explaining every flagged finding and checking each phase separately. Original
findings remain in the report. This is model-reviewed confirmation, not independent
or human verification. Missing evidence, provider errors, changed source/draft bytes,
and unmet size/reduction requirements cannot be overridden.

| Semantic status | Meaning |
| --- | --- |
| VERIFIED | Initial complete checks passed without material candidate losses. |
| CONFIRMED | Final model review resolved the flags, or the user explicitly accepted the draft. The report identifies which. |
| REVIEW_REQUIRED | Unresolved differences, uncertainty, missing evidence, or unmet token gates; draft retained. |
| ERROR | No usable draft was generated. |

Both VERIFIED and CONFIRMED publish an optimized copy; neither changes the source.
For REVIEW_REQUIRED drafts, the interactive CLI asks whether to accept after showing
the draft and report paths. **Yes sets CONFIRMED (by user)** and updates the report
and manifest while preserving the earlier model decision, warnings and unmet gates.
No or Enter keeps the draft unconfirmed. Without an interactive terminal, no consent
is inferred; `--accept-draft` explicitly authorizes user confirmation for that run.
User acceptance does not turn failed checks into passes. Changed source or draft
files still block publication. GEPA does not support this override.
The final step uses one call, with at most one schema retry, inside the existing budget.
The same environment helps compare answers but does not establish independent samples.

Evaluation does not demand a "why this matters" paragraph for every simple answer.
However, an explicitly required public output format remains part of the instruction
contract; silently deleting it is not meaning-preserving compression.

Drafts remain available when verification cannot pass. If the first model call fails
to produce a usable rewrite, Zen cannot guarantee a draft; a failure report explains
what happened. Filesystem or invalid-input errors can prevent output creation.
GEPA's opt-in acceptance policy is documented separately in the runtime guide.

## Current limits

- No tool execution, multi-turn workflow, or executable multi-file package evaluation.
- Semantic judgments are model-based proxies, not proof of equivalence or measured human comprehension.
- Token counts are local estimates, not billing or guaranteed future savings.
- Fresh application calls do not establish independent provider samples.
- Generated requirements and cases can miss behavior; acceptance is scoped to observed evidence.

See the [current limitations register](UNCONFIRMED_ASSUMPTIONS.md),
[example inputs](examples/README.md), and [full documentation](docs/documentation.md).

## Development

```powershell
uv run pytest -q
uv run zen selfcheck
uv run ruff check zen tests
```

Tests cover semantic workflow mechanics and opt-in GEPA integration with scripted
models, not universal quality, live provider reliability, or human-comprehension gains.