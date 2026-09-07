# Caveman review sample

[SKILL.md](SKILL.md) is the upstream code-review instruction artifact from
[JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman), revision
`0dd7ad6866949c6d69f04d7546c62a4f35c32a73`, path `skills/caveman-review/SKILL.md`.
Instruction text is unchanged; this local copy uses CRLF and a final newline,
whereas upstream uses LF without a final newline. The upstream licensing table assigns `skills/` to MIT;
the applicable copyright and permission notice are retained in [LICENSE](LICENSE).

The artifact formats code-review comments from supplied context, including exceptions
for security, architecture, and onboarding. It is a single-turn response sample,
not a benchmark of repository review, tools, linters, or GitHub integration.

Run from the workspace root:

```powershell
uv run zen --budget 32 optimize examples/in/caveman-review/SKILL.md --output-dir examples/out/caveman-review
```

This uses the normal semantic profile: three validation and three holdout cases,
the configured Terra/Sol/Luna defaults, and no implicit line cap. No favorable
outcome is promised. Review direct meaning findings, both phases, and separate
artifact/output token changes. A combined reduction alone does not prove quality.
Retain this MIT notice with redistributed source or derived drafts.

For a byte-exact upstream measurement when the reference checkout is available,
use `ref/caveman/skills/caveman-review/SKILL.md` as the input instead. The live
comparison uses that upstream path; the local copy is provided for portability.