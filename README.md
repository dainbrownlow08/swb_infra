# Switchboard conversational-styles

Re-analysis of *Determining Conversational Styles at Scale* (Switchboard-1): extract
per-utterance features, then test whether conversational style is two discrete types
(High-Involvement / High-Considerateness) or a single continuum. The short answer the
analysis reaches: a **unimodal, multi-dimensional continuum** — the paper's "two
styles" was an artifact of backchannels + a row-normalization bug.

## Layout

| Path | What's in it | Tracked? |
|------|--------------|----------|
| `analysis/` | The 6 PCA notebooks (numbered, in work order) + their guide. **Start at `analysis/README.md`.** | yes |
| `src/swb_extract/` | The extraction package: utterance pipeline + per-feature extractors. | yes |
| `scripts/` | Table builders (`build_tannen_features.py`, `build_merge_test.py`, `merge_features.py`). | yes |
| `tests/` | Unit tests, one per extractor. `pytest`. | yes |
| `docs/` | `AUDIT.md` (full audit), feature maps (`tannen_feature_map.md`, `tannen_features.txt`), `orthogonal_features_audit_jun19.md`, and the original paper PDF. | yes |
| `utterances_v2/` | Current extracted feature CSVs (`merge_test.csv`, `paper_aligned_standardized_PCA.csv`, `manifest.csv`, …). Regenerable; gitignored. | no |
| `tables/` | Switchboard corpus tables (`call_con_tab.csv` = side→caller map, `rating_tab.csv`, `caller_tab.csv`, …). | no |
| `corpus/` | NXT Switchboard annotations (gold dialog acts, prosody) — untapped validation data. | no |
| `audio/`, `utterances*/`, `swb_ms98_transcriptions_cleaned/` | Raw audio, sliced utterance wavs, transcripts. Large; gitignored. | no |
| `legacy/` | The original team's experiments and code, archived for reference. Not maintained. | no |

## Where to start

- **Understand the result:** `docs/AUDIT.md`, then `analysis/00_overview.ipynb`.
- **The notebooks:** `analysis/README.md` (build order + what each does).
- **The feature pipeline:** `src/swb_extract/` and `scripts/`.

Most large data directories are pulled separately and gitignored (see `.gitignore`);
the tracked surface is code, notebooks, and docs.
