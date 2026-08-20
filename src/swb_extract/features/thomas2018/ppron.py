"""ppron — rate of personal pronouns (Thomas et al. 2018 §3.2, "People").

"The rate of use of personal pronouns, including first- and second-, but not
third-person pronouns. As we needed a finer grain than the Linguistic Inquiry and Word
Count (LIWC) analysis included in MISC, this was based on our own list of pronouns."
The paper's list is unpublished; ours is the in-house 16-form 1st/2nd-person closed
list (pronoun_per_second v2: i/me/my/mine/myself, we/us/our/ours/ourselves,
you/your/yours/yourself/yourselves, y'all; contractions matched on their base). Rate =
pronouns per word, the LIWC convention the paper benchmarks its list against (LIWC's
category is literally named ``ppron``); a per-second variant would be the in-house
column. Per utterance this module writes the pronoun COUNT; the paper's ppron is
Σppron / Σwpu over the side (``aggregate``), never a mean of per-utterance ratios
(degenerate on short utterances — AUDIT.md §3 fix 3). Loading on involvement: +0.13;
the paper expected a weak signal because MISC tasks were assigned.

Output: utterances_v2/features/ppron.csv
Header: Utterance File Name,ppron
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ..inhouse.pronoun_per_second import PERSONAL_PRONOUNS, count_personal_pronouns
from ._io import Row, ratio_of_sums, write_feature_csv

FEATURE_NAME = "ppron"
HEADER = ("Utterance File Name", "ppron")

PRONOUNS = PERSONAL_PRONOUNS


def count_pronouns(text: str) -> int:
    return count_personal_pronouns(text)


def aggregate(rows: list[Row]) -> float | None:
    """Pronouns per word over the group: Σppron / Σwpu."""
    return ratio_of_sums(rows, "ppron", "wpu")


def write_ppron(manifest_csv: Path, output_csv: Path) -> int:
    return write_feature_csv(
        manifest_csv, output_csv, HEADER, lambda _rel, text: [str(count_pronouns(text))]
    )


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_ppron(manifest_path(out_root), out_root / "features" / f"{FEATURE_NAME}.csv")
    print(f"wrote {n} ppron rows")
    return 0
