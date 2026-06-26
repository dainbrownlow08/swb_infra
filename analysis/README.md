# Analysis notebooks

The re-analysis of *Determining Conversational Styles at Scale* (Switchboard:
per-utterance features → PCA → does a "two-style" HC/HI split exist, or is style a
continuum?). Notebooks are numbered in the order the work was done; each builds on
the one before.

| # | Notebook | What it does |
|---|----------|--------------|
| 00 | `00_overview.ipynb` | Narrative record of the whole re-analysis, stage by stage — read this first. |
| 01 | `01_replicate_legacy_matrix.ipynb` | First replication of the legacy `fix feature matrix` approach on the newly-extracted features. |
| 02 | `02_replicate_paper.ipynb` | Paper-aligned replication: Table 1's 14 features minus the 3 pitch features = 11. Reproduces the paper's structure. |
| 03 | `03_fix_scaling_and_backchannel_autopsy.ipynb` | **The pivot.** Replaces the paper's row-normalize with column-standardize, then shows the "two styles" bimodality is a **backchannel** artifact. **Writes the corrected matrix** the speaker-level notebooks consume. |
| 04 | `04_caller_level_volume.ipynb` | Caller-level test (volume features). One row per *person*, not per conversation side — see "Unit of analysis" below. |
| 05 | `05_caller_level_interactional.ipynb` | Caller-level test expanded with the interactional/timing features + varimax factor analysis. |

## Build order / dependencies

```
merge_test.csv ──► 03_fix_scaling ──► paper_aligned_standardized_PCA.csv ──► 04_caller_level_volume
merge_test.csv ─────────────────────────────────────────────────────────► 05_caller_level_interactional
                                                                            (+ tables/call_con_tab.csv)
```

Run **03 before 04** — 04 reads the standardized matrix that 03 produces. 01, 02,
05 read `merge_test.csv` directly and are independent of each other.

## Unit of analysis (important)

"Speaker" is the **caller**, not the conversation side. The filename gives a side
(`sw2001A` = conversation 2001, side A), but each caller appears across ~9 calls as
a different side. Notebooks 04/05 map side → `caller_no` via
`../tables/call_con_tab.csv` and aggregate at caller level (~493 people), so the
PCA / significance tests run on independent units rather than ~9×-pseudo-replicated
sides. (Earlier side-level versions over-counted n ~9× and produced a spurious k=2
BIC result and a spurious "NYC is most distinctive" effect.)

## Data locations

Notebooks read/write CSVs under `../utterances_v2/` (gitignored, regenerable) and
the corpus tables under `../tables/`. The notebooks themselves are version-controlled
here; their data is not.

## Running

```bash
jupyter nbconvert --to notebook --execute --inplace 00_overview.ipynb   # etc.
```
Uses the framework `python3` (pandas / sklearn / scipy). Run from inside `analysis/`.
