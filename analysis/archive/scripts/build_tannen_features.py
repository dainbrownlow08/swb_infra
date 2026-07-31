"""Build utterances_v2/tannen_features.csv by merging the Tannen-grounded
feature CSVs onto the manifest.

Inputs (under utterances_v2/):
- manifest.csv                                — Utterance File Name, Transcript
- features/topic_label.csv                    — Utterance File Name, Topic Label
- features/personal_focus_score.csv           — Utterance File Name, Personal Focus Score
- features/mutual_revelation_flag.csv         — Utterance File Name, Mutual Revelation Flag

Output:
- tannen_features.csv  (one row per utterance, joined on "Utterance File Name")

This is the Dimension 1 baseline. Subsequent dimensions will add more columns
to this file via the same join key.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "utterances_v2"
MANIFEST = ROOT / "manifest.csv"
FEATURES = ROOT / "features"
OUTPUT = ROOT / "tannen_features.csv"

TANNEN_FEATURE_CSVS = (
    "topic_label",
    "personal_focus_score",
    "mutual_revelation_flag",
)


def main() -> int:
    if not MANIFEST.is_file():
        raise SystemExit(f"missing manifest: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    print(f"loaded manifest: {df.shape}")

    for name in TANNEN_FEATURE_CSVS:
        path = FEATURES / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"missing feature CSV: {path}")
        feat = pd.read_csv(path)
        if feat.shape[1] < 2 or feat.columns[0] != "Utterance File Name":
            raise SystemExit(
                f"unexpected shape for {name}: {feat.shape[1]} cols, "
                f"first col {feat.columns[0]!r}"
            )
        before = df.shape
        df = df.merge(feat, on="Utterance File Name", how="left")
        added = [c for c in feat.columns if c != "Utterance File Name"]
        nan = {c: int(df[c].isna().sum()) for c in added}
        print(f"merged {name}: {before} -> {df.shape}  (added {added}, NaN={nan})")

    if df.shape[0] != pd.read_csv(MANIFEST).shape[0]:
        raise SystemExit(
            f"row-count drift: manifest had different row count than merged output"
        )

    df.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT}: {df.shape}")
    print(f"columns: {list(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
