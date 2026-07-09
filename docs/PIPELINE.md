# Pipeline & data contract

How feature data flows from raw extraction to the analysis notebooks, and the
procedure for adding/validating a feature. The goal of this layout is that the
trustworthy line (notebook 06 onward) **cannot accidentally run on stale or
unregistered data**.

## The one rule

There is exactly **one canonical table**: `utterances_v2/derived/features_table.csv`.
Every trustworthy/experimental notebook loads it through one function, and nothing
else. No notebook hardcodes a `read_csv` path to a feature table.

```python
from swb_extract.analysis import load_features_table
df = load_features_table(include="provisional")   # validated + unconfirmed (default)
```

## Data layers

| Layer | Path | Tracked? | Who writes it | Who reads it |
|---|---|---|---|---|
| **Input** | `utterances_v2/manifest.csv` | no (gitignored) | the extract pipeline | the table builder |
| **Input** | `utterances_v2/features/*.csv` | no | each feature extractor (`swb-extract features …`) | the table builder |
| **Trust metadata** | `docs/FEATURES.md` (parsed by `registry.py`) | **yes** | you (move rows between buckets) | the loader |
| **Derived** | `utterances_v2/derived/features_table.csv` | no (rebuildable) | `swb-extract table` | trustworthy notebooks via the loader |
| **Frozen** | `utterances_v2/merge_test.csv`, `paper_aligned_*` | no | the replication notebooks (01/02/03) | replication notebooks only |
| **Quarantine** | `utterances_v2/_archive/` | no | — | nobody (safe to delete) |

Inputs and derived data are gitignored (large, regenerable); **the code and the
registry are the version-controlled truth.** The canonical table is a pure
function of `manifest.csv` + `features/*.csv`, rebuilt in ~3 s.

## Two guards run on every load

`load_features_table()` refuses to hand you bad data:

1. **Stale guard** — if any `features/*.csv` or `manifest.csv` is newer than
   `features_table.csv`, it raises and tells you to `swb-extract table`. You can
   never silently analyze a table that predates a feature change.
2. **Registry guard** — if the table has a column not in the registry (or the
   registry names a column the table lacks), it raises. A new feature column
   cannot be used until it is registered with a trust status.

## The trust registry

`docs/FEATURES.md` is the single, human-editable source of truth for *what each
column is* and *whether we trust it* — a living document with **Trusted** / **WIP** /
**Deprecated** sections, one markdown-table row per column.
`src/swb_extract/registry.py` parses it; the section a row sits under is its trust
status, and each row also carries a `family` (volume | interactional | prosody |
tannen | meta):

- **Trusted** — correctness confirmed; analysis may rely on it.
- **WIP** — built and producing values, but not yet confirmed. Usable in
  experiments; the loader prints a warning naming every WIP column it hands you.
- **Deprecated** — do not use (kept for paper replication only, e.g. legacy `Turn Gap`).

**To change a feature's trust, move its row between sections in `docs/FEATURES.md`** —
that is the whole workflow.

Dashboard:

```python
python3 -c "import sys; sys.path.insert(0,'src'); from swb_extract import registry as R; print(R.summary())"
```

Select by trust/family in a notebook:

```python
load_features_table(include="validated")              # only trusted columns
load_features_table(include="provisional", family="interactional")
```

## Procedure: add (and validate) a feature

1. **Write the extractor** in `src/swb_extract/features/<name>.py` with a unit test
   (`tests/test_<name>.py`). It writes `utterances_v2/features/<name>.csv`, one row
   per manifest utterance, keyed by `Utterance File Name`, blanks (not 0) for
   "unmeasurable."
2. **Run it:** `swb-extract features <name>` → produces the sibling CSV.
3. **Rebuild the canonical table:** `swb-extract table`. The builder's per-row key
   assertions fail loudly on any misalignment — that is the corruption guard.
4. **Register the new column(s)** as rows in `docs/FEATURES.md` under **WIP**, with
   the extractor, `family`, and a one-line note. (The registry guard reminds you if
   you forget — `load_features_table` raises naming the unregistered column.)
5. **Validate**, then promote: confirm the extractor is correct (unit tests + a
   distribution sanity check in-notebook + ideally the NXT-gold check, audit §4C).
   When satisfied, **move its row to the Trusted section** of `docs/FEATURES.md`.

## Procedure: recompute the experimental notebook on the new table

Because the notebook loads the canonical table through `load_features_table`, the
loop is just:

```
swb-extract features <name>     # (if the extractor changed)
swb-extract table               # rebuild the single canonical table
# re-run the experimental notebook top-to-bottom
```

The stale guard guarantees step 2 happened: if you skip it, the notebook's first
`load_features_table()` raises instead of giving you old numbers. New validated
columns flow in automatically; provisional ones flow in with a warning.

## Replication tier (frozen — do not modify)

Notebooks 01/02 reproduce the legacy/paper result and read the **frozen**
`merge_test.csv` (and write their own `*_PCA_*.csv`). They are deliberately left on
the old data and must not be re-pointed at the canonical table — their job is to
reproduce the original (broken) result for the autopsy.
