# analysis/

Two notebooks matter; everything else is archived history.

| Notebook | Role |
|---|---|
| `08_measurement.ipynb` | **The living notebook.** The measurement-first analysis per `docs/NB08_MEASUREMENT_PLAN.md`: five combined trusted-space PCA coordinates (26 Trusted features, 487 callers, Horn K=5) + a theory-cited Tannen HI/HC projection score per caller, with validation; the type question demoted to a secondary test. All new analysis lands here. |
| `07_demoted_in_favor_of_5D.ipynb` | **The frozen evidence record** (formerly `07_final.ipynb`; closed 2026-07-29 — never edited, never re-run: ~55 min, and its recorded outputs are the evidence cited as "NB07 Step N"). Holds the audit-response arc as numbered Steps 11–24: trust adjudication, the NXT gold suite, overlap split, classifiers, the §4A dimensionality battery, clusterability, taxometrics (CCFI .151), power, the 288-spec multiverse, and Step 24's combined PCA. NB08 cites these numbers instead of recomputing them (ledger: `docs/AUDIT.md`). Planned later step: moves to `archive/` once NB08 stands alone. |

`validate_rising_terminal.py` — the gold-DA known-groups validation behind the 2026-07-29
rising-terminal semitone redesign (qy 2.48×, AUC .721; the failed `^d` arm adjudicated on
record). Kept here because `docs/FEATURES.md` cites it as the admission evidence for
`Terminal ST Slope`.

## archive/ — NB00–06, frozen replication artifacts

Moved here 2026-07-31 (git history preserves original paths). 00 overview · 01/02 legacy/paper
replication (reproduce the *broken* result for the autopsy) · 03 the pivot (row-norm →
column-standardize; bimodality = backchannel artifact) · 04/05 caller-level volume /
interactional+varimax · 06 Brizan revisions (FTO, dip, Horn, FDR). They read the frozen
`merge_test.csv`/`paper_aligned_*` line by `analysis/`-relative paths, which dangle one level
deeper — irrelevant by construction: frozen means never re-run. Their findings are compressed
into `docs/AUDIT.md` §1–§2.

## The trustworthy data line (07/08)

One canonical table, one loader, two guards:

```
swb-extract features <name>  ──►  utterances_v2/features/*.csv
swb-extract table            ──►  utterances_v2/derived/features_table.csv
load_features_table(...)     ──►  08_measurement.ipynb   (stale-data + registry guards on
                                                          every load; trust status lives in
                                                          docs/FEATURES.md)
```

## Unit of analysis (important)

"Speaker" is the **caller**, not the conversation side: each caller appears across ~9 calls
as different sides (`sw2001A` = conversation 2001, side A). NB07/NB08 map side → `caller_no`
via `../tables/call_con_tab.csv` and aggregate at caller level (493 mapped, 487 retained at
≥20 substantive utterances). Earlier side-level runs over-counted n ~9× and produced the
spurious k=2 BIC and "NYC most distinctive" results.

## Running

```bash
cd analysis && python3 -m jupyter nbconvert --to notebook --execute --inplace 08_measurement.ipynb
```

Framework `python3` (pandas / sklearn / scipy / diptest); cwd must be `analysis/` (notebooks
resolve `../tables/…`). NB08 targets a ~15–25 min clean run — background long runs, never pipe
nbconvert (a pipe masks its exit code). After any extractor change: `swb-extract features
<name> && swb-extract table`, then re-run 08 top-to-bottom and reconcile its close
(`docs/PIPELINE.md`). Big-edit rules (nbformat batches, backups + JSON diff, stubbed
smoke tests) are in `docs/NB08_MEASUREMENT_PLAN.md` §3. NB07 stays frozen.
