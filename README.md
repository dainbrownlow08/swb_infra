# Switchboard conversational-styles

## Layout

- `analysis/` — `08_measurement.ipynb` (**living**) · `07_demoted_in_favor_of_5D.ipynb` (**frozen record**) · `validate_rising_terminal.py` · `archive/` (NB00–06 + the frozen merge_test-era builders). Guide: `analysis/README.md`.
- `src/swb_extract/` — extractors + pipeline + the data layer: canonical-table builder (`swb-extract table`), trust-registry parser, the guarded loader, the §4A stats modules.
- `tests/` — `python3 -m pytest`.
- `docs/` — four living docs: `AUDIT.md` (canonical state: verdict, evidence ledger, roadmap) · `FEATURES.md` (trust registry, enforced at load) · `PIPELINE.md` (data contract) · `NB08_MEASUREMENT_PLAN.md` (current phase) — plus Tannen references, the paper PDF, and `archive/`.
- `utterances_v2/`, `tables/`, `corpus/`, `audio/`, `swb_ms98_transcriptions_cleaned/` — data: extracted CSVs + canonical table (regenerable), corpus tables, NXT gold, audio, transcripts. Gitignored.
- `legacy/` — the original team's code/data, inherited and untracked; reference only.

## Start here

- **State of everything:** `docs/AUDIT.md`.
- **Current analysis:** `analysis/08_measurement.ipynb`, built per `docs/NB08_MEASUREMENT_PLAN.md`.
- **Data layer:** `docs/PIPELINE.md` + `docs/FEATURES.md` (the loader refuses stale tables and unregistered columns).
