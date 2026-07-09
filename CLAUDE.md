# CLAUDE.md — Switchboard (conversational-styles re-analysis)

Project-specific working rules. Universal taste lives in `~/.claude/CLAUDE.md`.

## Keep `docs/AUDIT.md` honest

`docs/AUDIT.md` is the living audit of the repo **and** the roadmap for the re-analysis. Its
status markers carry the plan: **✅ done** (implemented *and in use*), **🟡 partial** (built but
not wired into analysis, or done at one level but not the rigorous version), **⬜ not started**.
The ✅-vs-🟡 line is the distinction that matters most here — most work stalls at 🟡 = built but
unused ("built ≠ in use"), and the gap is almost always the table/notebook integration layer.

- **At the end of every task, update `docs/AUDIT.md` to reflect reality.** For anything you
  touched: move the marker, rewrite its `↳ **Status:**` line to what is now true, and adjust the
  progress-dashboard counts if a marker changed. A fix is not "done" until the audit says what is
  actually true about it — in particular, whether it is merely *built* or genuinely *in use in the
  analysis*. Treat this as part of the task, not a chore after it.

- **Before the audit drives a sequencing or build decision, verify its load-bearing claims against
  the tree** — especially "X isn't built yet / no script does Y" markers, which are the ones that
  trigger redundant work. The doc lags the code (it is reviewed in batches, not continuously): a
  stale "done" only wastes a check, but a stale "not built" makes you rebuild something that already
  exists. Cheap tell: when a status line claims something is missing, `grep`/`ls` for it and compare
  file mtimes against the audit's own "last reviewed" date before acting. This is the doc-lag dual
  of the global "the checkout is not the world" rule.

  _Anchoring example (2026-06-29):_ §3.2 claimed "no from-source canonical-table builder exists"
  while `src/swb_extract/features_table.py` (+ the `swb_extract table` CLI command) — the literal
  deliverable, citing "AUDIT.md §3 fix 2" in its own docstring — had existed for 12 days. The marker
  was newer than the file but never mentioned it. Caught only by listing the tree, not by re-reading
  the audit.
