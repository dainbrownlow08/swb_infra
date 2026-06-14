"""Idempotently merge the new interactional feature CSVs onto merge_test.csv.

New features (utterances_v2/features/):
  latching_flag.csv            — Latching Flag
  question_flags.csv           — Question Flag, Echo Question Flag
  within_utterance_pauses.csv  — Within Pause {Total Sec, Count, Rate, Max ...}
  overlap.csv                  — Overlap {Duration Sec, Count, Onset Flag}
  rising_terminal.csv          — Rising Terminal Flag   (audio; may lag)

Plus a post-merge composite (needs rising_terminal + existing base columns):
  Machine Gun Question Score (0-4) and Machine Gun Question Flag
    A machine-gun question (Tannen, PDF p.112) = a question that is also short,
    fast-following, and high-pitched. Score gates on being a question (syntactic
    Question Flag OR prosodic Rising Terminal Flag), then adds one point each for
    short turn (token_count <= 4), fast follow (Turn Gap <= 0.5s), and above-
    median pitch. Non-questions score 0. Flag = score == 4 (all ingredients).
    The composite needs population pitch median, so it is computed here, not in a
    per-utterance extractor.

Idempotent: existing new-feature columns are dropped before re-merging, so this
can be run after the timing/text features land and again once the (slow) audio
rising_terminal finishes. Backs up merge_test.csv -> merge_test_backup.csv once.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "utterances_v2"
FEAT = ROOT / "features"
MERGE_TEST = ROOT / "merge_test.csv"
BACKUP = ROOT / "merge_test_backup.csv"
KEY = "Utterance File Name"

FEATURE_FILES = [
    "latching_flag.csv",
    "question_flags.csv",
    "within_utterance_pauses.csv",
    "overlap.csv",
    "rising_terminal.csv",
]

# Every column this script may add. Dropped up front so a re-run with a feature
# file temporarily absent removes its stale column rather than leaving it behind.
MANAGED_COLUMNS = [
    "Latching Flag",
    "Question Flag",
    "Echo Question Flag",
    "Within Pause Total Sec",
    "Within Pause Count",
    "Within Pause Rate",
    "Max Within Pause Sec",
    "Overlap Duration Sec",
    "Overlap Count",
    "Overlap Onset Flag",
    "Rising Terminal Flag",
    "Machine Gun Question Score",
    "Machine Gun Question Flag",
]

MG_MAX_TOKENS = 4
MG_MAX_GAP = 0.5


def main() -> int:
    if not MERGE_TEST.is_file():
        raise SystemExit(f"missing {MERGE_TEST}")
    df = pd.read_csv(MERGE_TEST)
    print(f"loaded merge_test: {df.shape}")

    if not BACKUP.exists():
        df.to_csv(BACKUP, index=False)
        print(f"backed up original -> {BACKUP.name}")

    # Drop ALL managed columns first so re-runs are fully idempotent (a feature
    # file that is temporarily absent loses its stale column instead of lingering).
    df = df.drop(columns=[c for c in MANAGED_COLUMNS if c in df.columns], errors="ignore")

    added: list[str] = []
    rising_merged = False
    for fn in FEATURE_FILES:
        path = FEAT / fn
        if not path.is_file():
            print(f"  skip (not present yet): {fn}")
            continue
        feat = pd.read_csv(path)
        cols = [c for c in feat.columns if c != KEY]
        df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
        before = df.shape
        df = df.merge(feat, on=KEY, how="left")
        nan = int(df[cols[0]].isna().sum())
        print(f"  merged {fn}: {before} -> {df.shape} (+{cols}, NaN_first={nan})")
        added += cols
        if fn == "rising_terminal.csv":
            rising_merged = True

    # --- machine-gun composite (only once the prosodic flag is actually merged) ---
    base_needed = {"Question Flag", "token_count", "Turn Gap", "pitch mean"}
    if rising_merged and base_needed.issubset(df.columns):
        q = pd.to_numeric(df["Question Flag"], errors="coerce").fillna(0)
        rt = pd.to_numeric(df["Rising Terminal Flag"], errors="coerce").fillna(0)
        is_q = ((q == 1) | (rt == 1)).astype(int)
        tok = pd.to_numeric(df["token_count"], errors="coerce")
        gap = pd.to_numeric(df["Turn Gap"], errors="coerce")
        pit = pd.to_numeric(df["pitch mean"], errors="coerce")
        pit_med = pit.median()
        short = (tok <= MG_MAX_TOKENS).astype(int)
        fast = (gap <= MG_MAX_GAP).astype(int)
        high = (pit >= pit_med).astype(int)
        score = is_q * (1 + short + fast + high)
        df = df.drop(
            columns=[
                c
                for c in ("Machine Gun Question Score", "Machine Gun Question Flag")
                if c in df.columns
            ],
            errors="ignore",
        )
        df["Machine Gun Question Score"] = score
        df["Machine Gun Question Flag"] = (score >= 4).astype(int)
        added += ["Machine Gun Question Score", "Machine Gun Question Flag"]
        print(
            f"  computed machine-gun composite "
            f"(questions={int(is_q.sum())}, flag=1: {int((score >= 4).sum())})"
        )
    else:
        print("  machine-gun composite skipped (rising_terminal not merged yet)")

    df.to_csv(MERGE_TEST, index=False)
    print(f"wrote {MERGE_TEST}: {df.shape}")
    print(f"new columns this run: {added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
