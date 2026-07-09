# Analysis notebooks

The re-analysis of *Determining Conversational Styles at Scale* (Switchboard:
per-utterance features → PCA → does a "two-style" HC/HI split exist, or is style a
continuum?). Notebooks are numbered in the order the work was done; **00–06 are
frozen** (replication artifacts — never edited, never re-pointed), and
**`07_final.ipynb` is the living notebook** where all new analysis lands.

| # | Notebook | What it does |
|---|----------|--------------|
| 00 | `00_overview.ipynb` | Narrative record of the whole re-analysis, stage by stage — read this first. |
| 01 | `01_replicate_legacy_matrix.ipynb` | First replication of the legacy `fix feature matrix` approach on the newly-extracted features. |
| 02 | `02_replicate_paper.ipynb` | Paper-aligned replication: Table 1's 14 features minus the 3 pitch features = 11. Reproduces the paper's structure. |
| 03 | `03_fix_scaling_and_backchannel_autopsy.ipynb` | **The pivot.** Replaces the paper's row-normalize with column-standardize, then shows the "two styles" bimodality is a **backchannel** artifact. Writes the corrected matrix the speaker-level notebooks consume. |
| 04 | `04_caller_level_volume.ipynb` | Caller-level test (volume features). One row per *person*, not per conversation side — see "Unit of analysis" below. |
| 05 | `05_caller_level_interactional.ipynb` | Caller-level test expanded with the interactional/timing features + varimax factor analysis. |
| 06 | `06_caller_level_volume_brizan_revisions_plus.ipynb` | Caller-level volume with the Brizan revisions: `Turn Gap` → **FTO**, Hartigan's dip, Horn's parallel analysis, FDR-corrected demographics. Frozen on the 03-era CSV. |
| 07 | `07_final.ipynb` | **The living notebook.** Rebuilds 06 on the **canonical table** via the guarded loader (reproduces 06 to within rounding), then accumulates the audit work as numbered Steps (Step 11 = trust adjudication; the §4A battery, gold suite, and taxometrics land next per `docs/Audit_July_2026_Paper_Submission_Min.md`). Steps insert before Step 10 + the Conclusion, which stay last; the Conclusion is re-reconciled at every landing. |

## Two data lines

**Frozen replication line (01–06):** reads `merge_test.csv` / `paper_aligned_*` built by
the (frozen) `scripts/` builders — kept byte-stable so the paper-replication notebooks
stay reproducible, known bugs and all.

```
merge_test.csv ──► 03_fix_scaling ──► paper_aligned_standardized_PCA.csv ──► 04, 06
merge_test.csv ──────────────────────────────────────────────────────────► 01, 02, 05
```

**Trustworthy line (07 onward):** one canonical table, one loader, two guards.

```
swb-extract features <name>  ──►  utterances_v2/features/*.csv
swb-extract table             ──►  utterances_v2/derived/features_table.csv
load_features_table(...)      ──►  07_final.ipynb        (stale-data + registry guards
                                                          on every load; trust status
                                                          lives in docs/FEATURES.md)
```

## Unit of analysis (important)

"Speaker" is the **caller**, not the conversation side. The filename gives a side
(`sw2001A` = conversation 2001, side A), but each caller appears across ~9 calls as
a different side. Notebooks 04–07 map side → `caller_no` via
`../tables/call_con_tab.csv` and aggregate at caller level (493 callers, 487 retained
at ≥20 substantive utterances), so the PCA / significance tests run on independent
units rather than ~9×-pseudo-replicated sides. (Earlier side-level versions
over-counted n ~9× and produced a spurious k=2 BIC result and a spurious "NYC is most
distinctive" effect.)

## Data locations

Notebooks read/write CSVs under `../utterances_v2/` (gitignored, regenerable) and the
corpus tables under `../tables/`. NB07 additionally refuses to run on a stale canonical
table — if any `features/*.csv` is newer than `derived/features_table.csv`, the loader
raises and tells you to `swb-extract table`. The notebooks themselves are
version-controlled here; their data is not.

## Running

```bash
cd analysis && python3 -m jupyter nbconvert --to notebook --execute --inplace 07_final.ipynb
```

Uses the framework `python3` (pandas / sklearn / scipy / diptest). Run from inside
`analysis/` — the notebooks resolve `../tables/…` relative to their own directory.
After any extractor change: `swb-extract features <name> && swb-extract table`, then
re-run 07 top-to-bottom and reconcile its Conclusion (see `docs/PIPELINE.md`).
