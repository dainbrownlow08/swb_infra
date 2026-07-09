# Switchboard conversational-styles

Re-analysis of *Determining Conversational Styles at Scale* (Switchboard-1): extract
per-utterance features, then test whether conversational style is two discrete types
(High-Involvement / High-Considerateness) or a single continuum. The short answer the
analysis reaches: a **unimodal, multi-dimensional continuum** — the paper's "two
styles" was an artifact of backchannels + a row-normalization bug. Current work
(see `docs/Audit_July_2026_Paper_Submission_Min.md`) is making that claim formally
airtight: the §4A modality battery, an NXT-gold validation suite, and taxometrics,
all landing in the living notebook `analysis/07_final.ipynb`.

## Layout

| Path | What's in it | Tracked? |
|------|--------------|----------|
| `analysis/` | The numbered notebooks: **00–06 are frozen** replication/history; **`07_final.ipynb` is the living notebook** where all new analysis lands. Guide: `analysis/README.md`. | yes |
| `src/swb_extract/` | The extraction package (per-feature extractors + utterance pipeline) **and the data layer**: `features_table.py` (canonical-table builder, `swb-extract table`), `registry.py` (parses the trust registry), `analysis.py` (`load_features_table` — the single guarded loader), `stats_modality.py` (the §4A test stack). | yes |
| `scripts/` | Legacy table builders (`build_merge_test.py`, …) — **frozen, replication-only**; superseded by `swb-extract table`. | yes |
| `tests/` | Unit tests (extractors, data layer, stats). `python3 -m pytest`. | yes |
| `docs/` | `AUDIT.md` (the living audit + execution plans), `FEATURES.md` (**the feature trust registry** — parsed and enforced at load), `PIPELINE.md` (data-layer contract + change loop), `Audit_July_2026_Paper_Submission_Min.md` (the current execution plan), feature maps, `orthogonal_features_audit_jun19.md`, the paper PDF. | yes |
| `utterances_v2/` | Extracted data: per-feature CSVs under `features/`, the **canonical table** `derived/features_table.csv` (the one table every trustworthy analysis loads), frozen `merge_test.csv`/`paper_aligned_*` for NB01–06, `_archive/`. Regenerable; gitignored. | no |
| `tables/` | Switchboard corpus tables (`call_con_tab.csv` = side→caller map, `rating_tab.csv`, `topic_tab.csv`, …). | no |
| `corpus/` | **NXT Switchboard gold** (642 conversations): dialog acts incl. backchannel/question tags, disfluency, syntax, prosody layers — the validation target for the gold suite. | no |
| `audio/`, `utterances*/`, `swb_ms98_transcriptions_cleaned/` | Raw audio, sliced utterance wavs, transcripts. Large; gitignored. | no |
| `legacy/` | The original team's experiments and code, archived for reference. Not maintained. | no |

## Where to start

- **Understand the result:** `docs/AUDIT.md` (verdict + status), then `analysis/00_overview.ipynb`.
- **The current analysis:** `analysis/07_final.ipynb` — the living notebook; every audit
  concern lands there as a numbered Step with a printed verdict.
- **The data layer:** `docs/PIPELINE.md` (how data flows, how to add a feature) and
  `docs/FEATURES.md` (which columns are Trusted / WIP / Deprecated — the loader refuses
  stale tables and unregistered columns).
- **What's being built right now:** `docs/Audit_July_2026_Paper_Submission_Min.md`.

Most large data directories are pulled separately and gitignored (see `.gitignore`);
the tracked surface is code, notebooks, and docs.
