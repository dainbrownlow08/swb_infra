# CLAUDE.md — Switchboard (conversational-styles re-analysis)

Project-specific working rules. Universal taste lives in `~/.claude/CLAUDE.md`.

## The doc contract (post-compression, 2026-07-31)

Four living documents, each with one job — everything superseded is verbatim in
`docs/archive/` and `analysis/archive/` (record, not guidance; never update it):

- `docs/AUDIT.md` — **canonical v2**: verdict, the NB07 evidence ledger, trust state,
  open items, roadmap. Keeps v1's §1–§5 numbering because code docstrings cite it.
- `docs/FEATURES.md` — the trust registry, machine-parsed (`registry.py`) and enforced by
  the loader. Moving a row between Trusted/WIP/Deprecated **is** the trust workflow.
- `docs/PIPELINE.md` — data-layer contract + the add-a-feature/recompute loop.
- `docs/NB08_MEASUREMENT_PLAN.md` — the current phase's execution plan (archived at phase end).

Notebooks: `analysis/08_measurement.ipynb` is the only living notebook;
`analysis/07_demoted_in_favor_of_5D.ipynb` is the frozen evidence record — never edit,
never re-run (~55 min; its recorded outputs are what "NB07 Step N" citations mean).

## Keep `docs/AUDIT.md` honest

Status markers carry the plan: **✅ done** (implemented *and in use*) · **🟡 partial** (built
but not wired into analysis) · **⬜ not started**. The ✅-vs-🟡 line matters most — work
stalls at "built ≠ in use," and the gap is almost always the notebook-integration layer.

- **At the end of every task, update `docs/AUDIT.md` to reflect reality** for anything you
  touched: move the marker, rewrite the status text to what is now true — in particular
  whether a thing is merely *built* or genuinely *in use*. This is part of the task, not a
  chore after it. Registry moves in `docs/FEATURES.md` count the same way.
- **Before the audit drives a build/sequencing decision, verify its load-bearing claims
  against the tree** — especially "X isn't built yet" markers, which trigger redundant work.
  The doc lags the code: a stale "done" wastes a check, but a stale "not built" makes you
  rebuild something that exists. Cheap tell: `grep`/`ls` for the claimed-missing thing and
  compare mtimes against the doc's review date. (Anchor case 2026-06-29: §3.2 claimed "no
  canonical-table builder exists" 12 days after `features_table.py` had landed — caught by
  listing the tree, not by re-reading the audit.)
