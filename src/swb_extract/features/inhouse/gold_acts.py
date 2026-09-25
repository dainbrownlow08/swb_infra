"""Gold dialog acts per utterance — the human-labelled pragmatics columns.

Tannen Ch.7 rows this serves (the ones no extractor could reach at the .8 bar — Question
Flag P .553, rising terminal P .15, Empath at chance): 3d "use of questions",
4b "information questions", 4a "echo questions as back-channel". The labels are the
SWBD-DAMSL dialog acts (Jurafsky, Shriberg & Biasca 1997) from two sources, joined to
ms98 utterances two ways and reconciled in ``analysis/validate_gold_acts.py``:

  nxt   — NXT Switchboard (642 convs): acts carry times through the terminals layer;
          ``nxt.align_to_utterances`` assigns each act to the utterance it overlaps most
          (≥ 50% of the shorter span). Preferred wherever available.
  swda  — the SwDA release (1,155 convs, no timestamps): word-sequence alignment of the
          two transcriptions (``swda.align_side``), with the SwDA caller↔ms98 channel
          mapping chosen per conversation (82 of 1,002 are A/B-swapped). Used for
          conversations NXT lacks.

An utterance's ``Gold Act`` is its TERMINAL act — the latest-starting act assigned to it
(nxt) / the act on its last matched word (swda) — because every downstream flag is about
what the utterance ends up doing. Columns (blank = no gold for that conversation, or the
utterance received no act):

  Gold Act                       — raw tag string, e.g. qy^d, bh, sd (every flag is
                                   derivable from it; parse with nxt.parse_tag)
  Gold DA Source                 — nxt | swda
  Gold Question Flag             — 1 if a question act: base ∈ {qy, qw, qo, qh, qr, qrr}
                                   or tag-question decoration ^g            (Tannen 3d)
  Gold Information Question Flag — qy / qw / qo / qr / qrr incl. their ^d declarative
                                   forms; excludes rhetorical qh and bh     (Tannen 4b)
  Gold Echo Question Flag        — bh (backchannel in question form, "oh really?"),
                                   or a question carrying the ^m mirror mark (Tannen 4a)

Only conversations present in the manifest are indexed, so a two-row test manifest
costs seconds and the corpus run minutes.

Output: utterances_v2/features/gold_acts.csv
Header: Utterance File Name,Gold Act,Gold DA Source,Gold Question Flag,Gold Information Question Flag,Gold Echo Question Flag
"""
from __future__ import annotations

import csv
from pathlib import Path

from ... import nxt, swda
from ...manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from ...transcripts import parse_transcript
from ._text import tokenize

FEATURE_NAME = "gold_acts"
HEADER = (
    "Utterance File Name",
    "Gold Act",
    "Gold DA Source",
    "Gold Question Flag",
    "Gold Information Question Flag",
    "Gold Echo Question Flag",
)

QUESTION_BASES = frozenset({"qy", "qw", "qo", "qh", "qr", "qrr"})
INFO_BASES = frozenset({"qy", "qw", "qo", "qr", "qrr"})

Key = tuple[int, str, int]
GoldIndex = dict[Key, tuple[str, str]]  # key → (terminal act tag, source)


def flags_for(tag: str) -> tuple[int, int, int]:
    """(question, information question, echo question) from a raw act tag."""
    bases, decos = nxt.parse_tag(tag)
    is_q = any(b in QUESTION_BASES for b in bases) or "^g" in decos
    info = any(b in INFO_BASES for b in bases)
    echo = ("bh" in bases) or (is_q and "^m" in decos)
    return int(is_q), int(info), int(echo)


def _trans_path(transcript_root: Path, conv: int, side: str) -> Path:
    return transcript_root / f"{conv:04d}"[:2] / f"{conv:04d}" / f"sw{conv:04d}{side}-ms98-a-trans.text"


def nxt_terminal_acts(
    conv: int, side: str, transcript_root: Path, xml_root: Path = nxt.NXT_XML
) -> dict[Key, str]:
    """Time-based join: utterance key → terminal act tag (latest-starting act assigned)."""
    tp = _trans_path(transcript_root, conv, side)
    if not tp.is_file():
        return {}
    utts = list(parse_transcript(tp))
    acts = [d for d in nxt.load_dialacts(conv, side, xml_root) if d.start is not None]
    best: dict[Key, tuple[float, str]] = {}
    for d, mi in zip(acts, nxt.align_to_utterances([(d.start, d.end) for d in acts],
                                                  [(u.start, u.end) for u in utts])):
        if mi is None:
            continue
        key = (conv, side, utts[mi].utt_num)
        if key not in best or d.start > best[key][0]:
            best[key] = (d.start, d.swbd_type)
    return {k: v[1] for k, v in best.items()}


def _ms98_tokens(transcript_root: Path, conv: int, side: str) -> list[tuple[str, int]]:
    tp = _trans_path(transcript_root, conv, side)
    if not tp.is_file():
        return []
    out: list[tuple[str, int]] = []
    for u in parse_transcript(tp):
        for tok in tokenize(u.text):
            t = swda.normalize_ms98_token(tok)
            if t:
                out.append((t, u.utt_num))
    return out


def swda_conversation_acts(
    conv: int, transcript_root: Path, swda_root: Path = swda.SWDA_ROOT
) -> tuple[dict[Key, str], dict[str, float], bool]:
    """Text join for both sides of one conversation.

    SwDA's caller labels sit on the other ms98 channel in 82 of 1,002 conversations (the
    known Switchboard A/B swap: identity match ≈ .1, swapped ≈ .95), so both mappings are
    aligned and the one with the higher total match ratio wins. Returns
    (utterance key → terminal act tag for sides at/above MIN_MATCH_RATIO,
     {ms98 side: match ratio}, swapped).
    """
    swda_acts = swda.load_acts(conv, swda_root)
    if not swda_acts:
        return {}, {}, False
    ms98 = {side: _ms98_tokens(transcript_root, conv, side) for side in "AB"}
    sw_tokens = {
        side: [(tok, i) for i, (_tag, text) in enumerate(acts)
               for tok in swda.normalize_swda_text(text)]
        for side, acts in swda_acts.items()
    }
    best = None
    for swapped, mapping in ((False, {"A": "A", "B": "B"}), (True, {"A": "B", "B": "A"})):
        ratios: dict[str, float] = {}
        res: dict[str, dict[Key, str]] = {}
        for sw_side, ms_side in mapping.items():
            by_utt, r = swda.align_side(sw_tokens.get(sw_side, []), ms98[ms_side])
            ratios[ms_side] = r
            res[ms_side] = {
                (conv, ms_side, utt): swda_acts[sw_side][idxs[-1]][0]
                for utt, idxs in by_utt.items()
            }
        score = sum(ratios.values())
        if best is None or score > best[0]:
            best = (score, swapped, ratios, res)
    _score, swapped, ratios, res = best
    acts: dict[Key, str] = {}
    for ms_side in "AB":
        if ratios.get(ms_side, 0.0) >= swda.MIN_MATCH_RATIO:
            acts.update(res[ms_side])
    return acts, ratios, swapped


def build_gold_index(
    convs: set[int],
    transcript_root: Path,
    swda_root: Path = swda.SWDA_ROOT,
    xml_root: Path = nxt.NXT_XML,
    prefer: str = "nxt",
) -> tuple[GoldIndex, dict]:
    """Terminal act per utterance for the given conversations; NXT first, SwDA for the rest.

    ``prefer="swda"`` forces the text join everywhere it exists (the validation script
    uses it to score the text join against the time join on the overlap).
    """
    nxt_convs = set(nxt.list_conversations(xml_root)) if xml_root.is_dir() else set()
    swda_convs = set(swda.list_conversations(swda_root)) if swda_root.is_dir() else set()
    idx: GoldIndex = {}
    stats = {"nxt_convs": 0, "swda_convs": 0, "swapped": 0, "low_ratio_sides": [], "ratios": []}
    for conv in sorted(convs):
        if conv in nxt_convs and prefer == "nxt":
            stats["nxt_convs"] += 1
            for side in "AB":
                for k, tag in nxt_terminal_acts(conv, side, transcript_root, xml_root).items():
                    idx[k] = (tag, "nxt")
        elif conv in swda_convs:
            stats["swda_convs"] += 1
            acts, ratios, swapped = swda_conversation_acts(conv, transcript_root, swda_root)
            stats["swapped"] += swapped
            for side, r in ratios.items():
                stats["ratios"].append(r)
                if r < swda.MIN_MATCH_RATIO:
                    stats["low_ratio_sides"].append((conv, side, round(r, 3)))
            for k, tag in acts.items():
                idx[k] = (tag, "swda")
    return idx, stats


def _row(rel: str, hit: tuple[str, str] | None) -> list[str]:
    if hit is None:
        return [rel, "", "", "", "", ""]
    tag, source = hit
    q, info, echo = flags_for(tag)
    return [rel, tag, source, str(q), str(info), str(echo)]


def write_gold_acts(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
    swda_root: Path = swda.SWDA_ROOT,
    xml_root: Path = nxt.NXT_XML,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(f"unexpected manifest header in {manifest_csv}: {header!r}")
        rels = [row[0] for row in reader if row]
    convs = set()
    for rel in rels:
        try:
            convs.add(parse_rel_path(rel)[0])
        except ValueError:
            pass
    idx, stats = build_gold_index(convs, transcript_root, swda_root, xml_root)
    n_labelled = n_q = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for rel in rels:
            try:
                hit = idx.get(parse_rel_path(rel))
            except ValueError:
                hit = None
            row = _row(rel, hit)
            writer.writerow(row)
            n_labelled += hit is not None
            n_q += row[3] == "1"
    ratios = stats["ratios"]
    print(
        f"  gold conversations: nxt={stats['nxt_convs']} swda={stats['swda_convs']} of "
        f"{len(convs)} in manifest; labelled utterances {n_labelled:,}/{len(rels):,} "
        f"({100 * n_labelled / max(len(rels), 1):.1f}%), questions {n_q:,}"
        + (f"; swda side match ratio median {sorted(ratios)[len(ratios) // 2]:.3f}, "
           f"A/B-swapped convs {stats['swapped']}, "
           f"{len(stats['low_ratio_sides'])} sides below {swda.MIN_MATCH_RATIO} blanked"
           if ratios else "")
    )
    return len(rels)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_gold_acts(
        manifest_path(out_root),
        out_root / "features" / f"{FEATURE_NAME}.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} gold act rows")
    return 0
