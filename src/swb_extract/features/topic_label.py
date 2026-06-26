"""Topic label per utterance, derived from LDC tables.

Source of truth (shipped by LDC as part of Switchboard-1):
- tables/conv_tab.csv  : per-conversation ivi_no (topic ID)
- tables/topic_tab.csv : ivi_no → topic_description (e.g. "CLOTHING AND DRESS")

Tannen reference: Ch.4 "Personal vs. Impersonal Topics" (PDF pp. 90-103).
The topic label is a conversation-level covariate — all utterances in a call
share the same label. It is intended for control / stratification / sanity-check
use, NOT as a PCA input. See plan file for rationale.

Output: utterances_v2/features/topic_label.csv
Header: Utterance File Name,Topic Label
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path

FEATURE_NAME = "topic_label"
HEADER = ("Utterance File Name", "Topic Label")

# Columns in conv_tab.csv (per tables/conv_doc.txt)
CONV_COL_NO = 0
CONV_COL_IVI = 4
# Columns in topic_tab.csv (per tables/topic_doc.txt)
TOPIC_COL_DESC = 0
TOPIC_COL_IVI = 1


def _strip_csv_field(s: str) -> str:
    return s.strip().strip('"')


def load_topic_tab(path: Path) -> dict[int, str]:
    """ivi_no → topic_description (e.g. {303: "CLOTHING AND DRESS"})."""
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) <= TOPIC_COL_IVI:
                continue
            desc = _strip_csv_field(row[TOPIC_COL_DESC])
            try:
                ivi = int(_strip_csv_field(row[TOPIC_COL_IVI]))
            except ValueError:
                continue
            out[ivi] = desc
    return out


UNKNOWN_TOPIC = "UNK"


def load_conv_topics(path: Path) -> dict[int, int | str]:
    """conversation_no → ivi_no (int) or "UNK" if the topic is unknown.

    LDC's conv_tab.csv has some rows with ivi_no=UNK (e.g. call 3178).
    Those calls are preserved with the sentinel "UNK" rather than dropped,
    so downstream code can emit "UNK" for utterances in those calls.
    """
    out: dict[int, int | str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) <= CONV_COL_IVI:
                continue
            try:
                cid = int(_strip_csv_field(row[CONV_COL_NO]))
            except ValueError:
                continue
            ivi_str = _strip_csv_field(row[CONV_COL_IVI])
            try:
                out[cid] = int(ivi_str)
            except ValueError:
                out[cid] = UNKNOWN_TOPIC
    return out


def build_topic_index(conv_path: Path, topic_path: Path) -> dict[int, str]:
    """Compose: call_id → topic_description (or "UNK" if unresolved)."""
    topics = load_topic_tab(topic_path)
    convs = load_conv_topics(conv_path)
    out: dict[int, str] = {}
    for cid, ivi in convs.items():
        if isinstance(ivi, str):
            out[cid] = UNKNOWN_TOPIC
        else:
            out[cid] = topics.get(ivi, UNKNOWN_TOPIC)
    return out


def write_topic_labels(
    manifest_csv: Path,
    output_csv: Path,
    topic_index: dict[int, str],
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_unk = 0
    n_missing = 0
    with open(manifest_csv, encoding="utf-8", newline="") as fin, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest_csv}: {header!r}"
            )
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for row in reader:
            if not row:
                continue
            rel = row[0]
            call_id, _side, _utt = parse_rel_path(rel)
            desc = topic_index.get(call_id)
            if desc is None:
                desc = UNKNOWN_TOPIC
                n_missing += 1
            elif desc == UNKNOWN_TOPIC:
                n_unk += 1
            writer.writerow([rel, desc])
            n += 1
    if n_unk or n_missing:
        print(
            f"  topic_label: {n_unk} utterances with UNK ivi_no, "
            f"{n_missing} calls not in conv_tab"
        )
    return n


def run(args) -> int:
    conv = Path(args.conv_tab)
    topic = Path(args.topic_tab)
    out_root = Path(args.out_root)
    idx = build_topic_index(conv, topic)
    n = write_topic_labels(
        manifest_path(out_root),
        out_root / "features" / "topic_label.csv",
        idx,
    )
    print(f"wrote {n} topic label rows for {len(idx)} conversations")
    return 0
