"""Machine-gun question composite — Tannen's rapid-fire question device.

Tannen ref: Ch.4 (PDF p.112): machine-gun questions are (a) questions, fired
(b) fast, with (c) reduced syntax, at (d) marked high pitch. A canonical
High-Involvement floor device.

Replaces the ad-hoc composite previously computed inside
``scripts/build_merge_test.py`` (AUDIT.md §3 fix 4), which had three defects:
"high pitch" meant above the WHOLE-POPULATION median (≈ a female-speaker
indicator, since female F0 sits above the pooled median); "fast follow" meant
``Turn Gap <= 0.5 s``, which every backchannel-polluted negative gap satisfied
for free; and rising-terminal NaN was filled with 0, conflating "unmeasurable"
with "not rising".

This extractor is a pure composite over sibling feature CSVs (no audio):
question_flags, rising_terminal, token_count, pitch, fto — all read in
manifest order with per-row key assertions, so it must run after them.

Ingredients per utterance:
  intent   — Question Flag == 1 (syntactic) OR Rising Terminal Flag == 1
             (prosodic). The intent gate must be determinate: if the syntax
             says no and the prosody was unmeasurable, we cannot rule out a
             declarative-form question, so the whole composite is EMPTY
             rather than 0.
  brevity  — token_count <= MAX_TOKENS (reduced syntax)
  speed    — Onset Gap Sec <= MAX_ONSET_SEC. Onset is FTO for turn-initial
             utterances, own-series gap for continuations (rapid-fire
             second/third questions), and negative for interjections fired
             into the other's ongoing speech. Unmeasurable onset
             (conversation-initial) earns no point.
  pitch    — utterance pitch mean >= the SPEAKER-SIDE's own
             HIGH_PITCH_PCTL-th percentile, i.e. marked relative to the
             speaker's baseline, not the population's. Sides need
             >= MIN_SIDE_PITCH_N pitched utterances to have a baseline;
             missing pitch or baseline earns no point (conservative).

  Machine Gun Question Score = intent * (1 + brevity + speed + pitch) ∈ {0..4}
  Machine Gun Question Flag  = 1 iff score == 4 (all ingredients), else 0
  Both empty when: utterance is a backchannel (not floor-directed talk) or
  the intent gate is indeterminate (see above).

Output: utterances_v2/features/machine_gun_question.csv
Header: Utterance File Name,Machine Gun Question Score,Machine Gun Question Flag
"""
from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path

FEATURE_NAME = "machine_gun_question"
HEADER = (
    "Utterance File Name",
    "Machine Gun Question Score",
    "Machine Gun Question Flag",
)

MAX_TOKENS = 4          # "reduced syntax"
MAX_ONSET_SEC = 0.5     # "fast" onset (FTO / series gap / overlap)
HIGH_PITCH_PCTL = 75    # marked-high relative to the speaker-side's own F0
MIN_SIDE_PITCH_N = 10   # minimum pitched utterances for a side baseline

_SIDE_RE = re.compile(r"(sw\d+[AB])")


class CompositeInputError(RuntimeError):
    pass


def _side_of(rel: str) -> str | None:
    m = _SIDE_RE.search(rel)
    return m.group(1) if m else None


def _open_feature(features_dir: Path, name: str, expected_cols: tuple[str, ...]):
    path = features_dir / f"{name}.csv"
    if not path.is_file():
        raise CompositeInputError(
            f"missing input {path} — run `swb-extract features {name}` first"
        )
    f = open(path, encoding="utf-8", newline="")
    reader = csv.reader(f)
    header = next(reader, None)
    if header is None or tuple(header) != expected_cols:
        f.close()
        raise CompositeInputError(
            f"{path.name}: unexpected header {header!r} (need {expected_cols}) — "
            f"re-run its extractor"
        )
    return f, reader


def build_side_pitch_baselines(features_dir: Path) -> dict[str, float]:
    """Per conversation-side HIGH_PITCH_PCTL-th percentile of pitch mean."""
    f, reader = _open_feature(
        features_dir, "pitch", ("Utterance File Name", "pitch mean", "pitch std", "pitch range")
    )
    by_side: dict[str, list[float]] = {}
    with f:
        for row in reader:
            if not row or not row[1]:
                continue
            side = _side_of(row[0])
            if side is not None:
                by_side.setdefault(side, []).append(float(row[1]))
    baselines: dict[str, float] = {}
    for side, vals in by_side.items():
        if len(vals) >= MIN_SIDE_PITCH_N:
            q = statistics.quantiles(vals, n=100, method="inclusive")
            baselines[side] = q[HIGH_PITCH_PCTL - 1]
    return baselines


def compose(
    qflag: str,
    rt_flag: str,
    token_count: str,
    onset: str,
    bc_flag: str,
    pitch_mean: str,
    side_baseline: float | None,
) -> tuple[int | None, int | None]:
    """(score, flag) from raw CSV cells; (None, None) = empty."""
    if bc_flag == "1":
        return None, None
    syntactic = qflag == "1"
    if not syntactic and rt_flag == "":
        return None, None  # prosody unmeasurable: intent indeterminate
    intent = syntactic or rt_flag == "1"
    if not intent:
        return 0, 0
    score = 1
    if token_count and int(token_count) <= MAX_TOKENS:
        score += 1
    if onset and float(onset) <= MAX_ONSET_SEC:
        score += 1
    if pitch_mean and side_baseline is not None and float(pitch_mean) >= side_baseline:
        score += 1
    return score, 1 if score == 4 else 0


def write_machine_gun(
    manifest_csv: Path, features_dir: Path, output_csv: Path
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    baselines = build_side_pitch_baselines(features_dir)
    print(f"  side pitch baselines: {len(baselines)}")

    fq, q_reader = _open_feature(
        features_dir, "question_flags",
        ("Utterance File Name", "Question Flag", "Echo Question Flag"),
    )
    fr, rt_reader = _open_feature(
        features_dir, "rising_terminal",
        ("Utterance File Name", "Rising Terminal Flag", "Terminal F0 Slope"),
    )
    ft, tok_reader = _open_feature(
        features_dir, "token_count", ("Utterance File Name", "token_count")
    )
    ff, fto_reader = _open_feature(
        features_dir, "fto",
        ("Utterance File Name", "FTO Sec", "Onset Gap Sec", "Turn Initial Flag",
         "Backchannel Flag", "Interjection Flag"),
    )
    fp, p_reader = _open_feature(
        features_dir, "pitch",
        ("Utterance File Name", "pitch mean", "pitch std", "pitch range"),
    )

    n = 0
    n_defined = 0
    n_flag = 0
    with fq, fr, ft, ff, fp, open(
        manifest_csv, encoding="utf-8", newline=""
    ) as fin, open(output_csv, "w", encoding="utf-8", newline="") as fout:
        man = csv.reader(fin)
        header = next(man, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest_csv}: {header!r}"
            )
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for row in man:
            if not row:
                continue
            rel = row[0]
            cells = []
            for name, reader in (
                ("question_flags", q_reader),
                ("rising_terminal", rt_reader),
                ("token_count", tok_reader),
                ("fto", fto_reader),
                ("pitch", p_reader),
            ):
                frow = next(reader, None)
                if frow is None or frow[0] != rel:
                    got = frow[0] if frow else "<end of file>"
                    raise CompositeInputError(
                        f"{name}.csv out of sync at manifest row {n + 1}: "
                        f"expected {rel!r}, got {got!r} — re-run its extractor"
                    )
                cells.append(frow)
            qrow, rtrow, tokrow, ftorow, prow = cells
            score, flag = compose(
                qflag=qrow[1],
                rt_flag=rtrow[1],
                token_count=tokrow[1],
                onset=ftorow[2],
                bc_flag=ftorow[4],
                pitch_mean=prow[1],
                side_baseline=baselines.get(_side_of(rel) or ""),
            )
            writer.writerow([
                rel,
                "" if score is None else str(score),
                "" if flag is None else str(flag),
            ])
            n += 1
            n_defined += score is not None
            n_flag += flag == 1
    print(
        f"  defined={n_defined} ({100 * n_defined / n:.1f}%), "
        f"flagged={n_flag} ({100 * n_flag / max(n_defined, 1):.2f}% of defined)"
    )
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_machine_gun(
        manifest_path(out_root),
        out_root / "features",
        out_root / "features" / "machine_gun_question.csv",
    )
    print(f"wrote {n} machine gun question rows")
    return 0
