"""Append word_rate and token_count columns to utterances_v2/merge_test.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "utterances_v2"
MERGE = ROOT / "merge_test.csv"
FEATURES = ROOT / "features"


def main() -> int:
    df = pd.read_csv(MERGE)
    before_cols = list(df.columns)
    print(f"loaded {MERGE.name}: {df.shape}")

    for name in ("word_rate", "token_count"):
        feat = pd.read_csv(FEATURES / f"{name}.csv")
        if name in df.columns:
            df = df.drop(columns=[name])
        df = df.merge(feat, on="Utterance File Name", how="left")
        nan_count = df[name].isna().sum()
        print(f"merged {name}: NaN rows = {nan_count}")

    df.to_csv(MERGE, index=False)
    new_cols = [c for c in df.columns if c not in before_cols]
    print(f"wrote {MERGE.name}: {df.shape}, new columns = {new_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
