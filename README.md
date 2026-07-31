# Switchboard conversational-styles

Re-analysis of *Determining Conversational Styles at Scale* (Switchboard-1). The paper's
"two styles" (High-Involvement / High-Considerateness) was an artifact of backchannels + a
row-normalization bug; the corrected result is a **unimodal, multi-dimensional continuum**,
formally supported by the full dimensionality battery, an NXT-gold validation suite,
taxometrics, and a 288-spec multiverse — all recorded in the frozen evidence notebook
`analysis/07_demoted_in_favor_of_5D.ipynb`. Current work is the **measurement phase**
(`docs/NB08_MEASUREMENT_PLAN.md`): five combined-PCA style coordinates + a continuous
theory-derived Tannen HI/HC score per caller, landing in the living notebook
`analysis/08_measurement.ipynb`.

The repo is deliberately compressed around those two notebooks (2026-07-31): 07 is the
citable record, 08 is the analysis; everything superseded lives in `analysis/archive/` and
`docs/archive/`.

## Layout

| Path | What's in it | Tracked? |
|------|--------------|----------|
| `analysis/` | `08_measurement.ipynb` (**living**), `07_demoted_in_favor_of_5D.ipynb` (**frozen record**), the rising-terminal validation script, and `archive/` (NB00–06). Guide: `analysis/README.md`. | yes |
| `src/swb_extract/` | The extraction package (per-feature extractors + utterance pipeline) **and the data layer**: `features_table.py` (canonical-table builder, `swb-extract table`), `registry.py` (parses the trust registry), `analysis.py` (`load_features_table` — the single guarded loader), `stats_modality.py` + `taxometrics.py` (the §4A test stack). | yes |
| `scripts/` | Legacy table builders — **frozen, replication-only**; superseded by `swb-extract table`. | yes |
| `tests/` | Unit tests (extractors, data layer, stats). `python3 -m pytest`. | yes |
| `docs/` | Four living docs — `AUDIT.md` (canonical state: verdict, evidence ledger, roadmap), `FEATURES.md` (**trust registry**, machine-parsed + enforced at load), `PIPELINE.md` (data contract), `NB08_MEASUREMENT_PLAN.md` (current phase) — plus the Tannen theory references, the paper PDF, and `archive/`. | yes |
| `utterances_v2/` | Extracted data: per-feature CSVs, the **canonical table** `derived/features_table.csv`, frozen `merge_test.csv`/`paper_aligned_*` for the archived notebooks. Regenerable; gitignored. | no |
| `tables/` | Switchboard corpus tables (`call_con_tab.csv` = side→caller map, `rating_tab.csv`, `topic_tab.csv`, …). | no |
| `corpus/` | **NXT Switchboard gold** (642 conversations): dialog acts incl. backchannel/question tags, disfluency, prosody — the gold-suite validation target. | no |
| `audio/`, `utterances*/`, `swb_ms98_transcriptions_cleaned/` | Raw audio, sliced utterance wavs, transcripts. Large; gitignored. | no |
| `legacy/` | The original team's experiments and code, archived for reference. Not maintained. | no |

## Where to start

- **The state of everything:** `docs/AUDIT.md` — verdict, the NB07 evidence ledger, trust
  state, and the roadmap, on one page.
- **The current analysis:** `analysis/08_measurement.ipynb`, built per
  `docs/NB08_MEASUREMENT_PLAN.md`; it cites the frozen record rather than recomputing it.
- **The data layer:** `docs/PIPELINE.md` (flow + add-a-feature loop) and `docs/FEATURES.md`
  (Trusted / WIP / Deprecated — the loader refuses stale tables and unregistered columns).

Most large data directories are pulled separately and gitignored (see `.gitignore`);
the tracked surface is code, the two notebooks, and the docs.
