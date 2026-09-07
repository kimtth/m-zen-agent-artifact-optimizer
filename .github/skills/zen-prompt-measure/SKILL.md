---
name: zen-prompt-measure
description: Use when optimizing a GitHub Copilot customization artifact with the Zen CLI, running zen optimize/detect/selfcheck, or interpreting a Zen report. Do not use for generic prompt rewriting, hooks, mcp.json, or automatic candidate application.
---

# Optimize a Copilot customization artifact

Goal: produce a compact semantic rewrite with a usable draft and bounded verification,
not a guarantee of identical outputs or universal meaning preservation.

Workflow:
1. Run `zen detect PATH` for `AGENTS.md`, `copilot-instructions.md`, `*.instructions.md`, `*.prompt.md`, `*.agent.md`, or `SKILL.md`.
2. Choose the actual target model and a strong rewrite/judge model.
3. Set `--budget` before an unfamiliar run. Semantic mode defaults to a small budget;
   `--max-metric-calls` is for the opt-in GEPA engine only.
4. Run `zen selfcheck` after optimizer code changes.
5. Run `zen optimize PATH`.
6. Read the report, differences and errors. Get this run's paths from the CLI or
  manifest, not the path-free report. Review the draft
  manually; never apply it automatically. For REVIEW_REQUIRED, the CLI asks whether
  to accept. Yes sets CONFIRMED by user and records the earlier model assessment in
  the report. Never supply --accept-draft without explicit user authorization;
  acceptance does not mean the model checks passed or replace the source.

The default semantic engine reads the full body and rewrites it, consolidating
repetition while preserving actions, constraints, conditions, exceptions, priorities,
language, and required public formats. It saves a usable draft before evaluation,
then uses direct semantic review, bounded correction, and source-based task comparisons.
No supplied labeled dataset is required. GEPA search remains opt-in with `--engine gepa`.

Semantic interpretation:
- Verification is separate from draft availability. Incomplete evidence, important
  meaning loss, or insufficient reduction requires review; a saved draft is retained.
- Complete but disputed evidence gets one final adjudication inside the shared budget
  (one schema retry, no provider retry). Model-origin CONFIRMED means the final model resolved every
  initial flag and found no material regression in either phase; it is not VERIFIED
  or human approval. Keep the original findings visible. Missing/error evidence,
  source/draft changes and deterministic token/line gates cannot be overridden.
- User-origin CONFIRMED means explicit acceptance of a REVIEW_REQUIRED draft, even
  with unmet model gates. Keep the original model decision, findings and errors in
  the report. Source and frozen-file integrity checks still block publication.
- Shared baseline defects are not automatically regressions, but cannot excuse lost
  instructions. Final review reads frozen holdout evidence without changing the draft.
- Harmless wording/order/length differences are not regressions. Missing important
  facts, changed exceptions, or added requirements can be.
- Exact answer quotations and unnecessary reasoning/verification narration are not
  mandatory evaluation conditions. Explicit source requirements cannot be silently
  removed just because they are verbose.
- Validation and holdout remain separate; freeze the candidate before holdout and
  never use holdout findings as repair feedback.
- Malformed model responses have bounded retries; errors never become successful
  scores. First-generation failure can leave no draft; filesystem failures can
  prevent a report. Do not promise unconditional generation.
- Semantic judgments are model proxies, not proof of equivalence or human understanding.
  Tokens are local estimates, not billing. Fresh calls do not establish independent samples.
- Source hash checks and frozen metadata protect the original. Use current report
  paths, not a stale candidate from an older run.

Semantic mode has no default hard line cap. `--aggressive LINES` or
`--aggressive PERCENT%` requests one; it is not evidence of preserved meaning.

Optional `--focus task` / `--focus communication` requires exact comment-delimited
sections; it does not infer them.

GEPA interpretation: `ACCEPT` still requires complete aligned error-free trials,
zero candidate critical failures, at most a 5pp drop in independent behavior/reader
trial rates, at least one behavior and reader pass per case, non-increasing
artifact/median output/reader tokens, and at least 3% combined reduction.
Its single-trial validation and small samples can make that tolerance ineffective.
GEPA `REJECT`/`INCONCLUSIVE` do not publish candidates. Its default line cap is 50%;
`--compare-concision` is GEPA-only measurement, never a selected artifact or GEPA input.
See [the runtime guide](../../../docs/documentation.md).

Limits: one artifact, response-only single-turn cases, no repository-changing tools, frozen metadata, no hidden reasoning, and no automatic source replacement.
