---
name: zen-prompt-measure
description: Use when optimizing a GitHub Copilot customization artifact with the Zen CLI, running zen optimize/detect/selfcheck, or interpreting a Zen report. Do not use for generic prompt rewriting, hooks, mcp.json, or automatic candidate application.
---

# Optimize a Copilot customization artifact

Goal: reduce artifact input and agent output without reducing behavior or reader understanding.

Workflow:
1. Run `zen detect PATH` for `AGENTS.md`, `copilot-instructions.md`, `*.instructions.md`, `*.prompt.md`, `*.agent.md`, or `SKILL.md`.
2. Choose the actual target model and a strong contract/judge/reflection model.
3. Set both `--budget` and `--max-metric-calls` before an unfamiliar run.
4. Run `zen selfcheck` after optimizer code changes.
5. Run `zen optimize PATH`.
6. Read the concise report and detailed run data. Review an accepted candidate manually; never apply it automatically.

Zen generates its own source-grounded behavior contract and multilingual evaluation cases. Do not create or request a reviewed `PATH.tests.json` file. The old artifact-kind rule table and DELETE/REWRITE/KEEP loop are not part of this implementation.

Interpretation:
- `ACCEPT` means the sealed holdout gate found no quality regression, no token-measure regression, and at least 3% combined communication-token reduction.
- `REJECT` means the source was retained because validation, holdout quality, token measurements, or the minimum reduction failed.
- Generated rules are valid only when they cite exact artifact-body evidence.
- A smaller candidate is never sufficient by itself.

Limits: one artifact, response-only single-turn cases, no repository-changing tools, frozen metadata, no hidden reasoning, and no automatic source replacement.
