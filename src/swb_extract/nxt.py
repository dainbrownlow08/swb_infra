"""NXT Switchboard gold-annotation access — dialog acts, terminals, disfluency.

The shared parsing layer for the gold-validation suite (AUDIT.md §4C12 / D3;
submission plan T4/Delta 7). Verified on-disk coverage (2026-07-09): 1,284 side
files per layer = 642 conversations, at
``corpus/nxt_switchboard_ann/xml/{dialAct,terminals,disfluency}/sw####.{A,B}.*.xml``.

Layers
------
- ``terminals/``  — word tokens carrying ``nite:start``/``nite:end`` times
  (``punc``/``sil``/``trace`` elements carry no times and are skipped).
- ``dialAct/``    — SWBD-DAMSL dialog acts. ``<da>`` elements carry no times;
  spans are resolved through their child terminal references.
- ``disfluency/`` — ``<disfluency>`` groups with ``<reparandum>``/``<repair>``
  terminal references (full 1,284-file coverage — the repetition
  de-conflation layer).

Tag grammar
-----------
``swbdType`` may join multiple acts with ``,`` and stacks decorations:
``^d`` declarative, ``^g`` tag-question, ``^m`` mirror/repeat-other,
``^r`` repeat-self, ``^e`` elaboration, ``^t`` about-task, ``^c``
about-communication, ``^h`` hold, ``^2`` collaborative completion,
``^q``/``(^q)`` quotation, ``@`` transcription problem, ``*`` segmentation
mark. A pure-decoration act keeps its innermost marker as the base
(``parse_tag("^2") == (("^2",), frozenset())``). **Never exact-string-match a
swbdType** — the corpus carries compounds like ``qy^g^t``, ``sd(^q)*`` and
``ba,fe``; always go through :func:`parse_tag` (Delta 7's pre-registration
discipline).

References: Jurafsky, Shriberg & Biasca 1997 (SWBD-DAMSL coders manual,
TR 97-02); Calhoun et al. 2010, *LREC Journal* 44(4) — the NXT resource.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NXT_XML = REPO_ROOT / "corpus" / "nxt_switchboard_ann" / "xml"

NITE = "{http://nite.sourceforge.net/}"

_HREF_ID_RE = re.compile(r"#id\(([^)]+)\)")
_DECO_TAIL_RE = re.compile(r"(\(\^q\)|\^[a-z0-9]+|@|\*)$")


def _side_file(layer: str, conv: int, side: str, xml_root: Path) -> Path:
    return xml_root / layer / f"sw{conv}.{side}.{layer}.xml"


def list_conversations(xml_root: Path = NXT_XML) -> list[int]:
    """Conversation ids with gold dialog acts (642 on the full corpus)."""
    return sorted({int(p.name[2:6]) for p in (xml_root / "dialAct").glob("sw*.dialAct.xml")})


def parse_tag(swbd_type: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """Split a raw ``swbdType`` into base act tag(s) and decorations.

    ``"qy^d^t"`` → ``(("qy",), {"^d", "^t"})``; ``"sd(^q)"`` → ``(("sd",),
    {"(^q)"})``; ``"ba,fe"`` → ``(("ba", "fe"), ∅)``; a decoration-only act
    keeps its innermost marker as the base: ``"^2"`` → ``(("^2",), ∅)``.
    """
    bases: list[str] = []
    decorations: set[str] = set()
    for part in swbd_type.split(","):
        part = part.strip()
        stripped: list[str] = []
        while part:
            m = _DECO_TAIL_RE.search(part)
            if not m:
                break
            stripped.append(m.group(1))
            part = part[: m.start()]
        if part:
            bases.append(part)
            decorations.update(stripped)
        elif stripped:
            bases.append(stripped[-1])          # innermost marker is the act
            decorations.update(stripped[:-1])
    return tuple(bases), frozenset(decorations)


def _href_ids(el) -> list[str]:
    ids: list[str] = []
    for child in el.iter(f"{NITE}child"):
        ids.extend(_HREF_ID_RE.findall(child.get("href", "")))
    return ids


def load_terminal_times(
    conv: int, side: str, xml_root: Path = NXT_XML
) -> dict[str, tuple[float, float]]:
    """Terminal id → (start, end) seconds; elements without numeric times skipped."""
    root = ET.parse(_side_file("terminals", conv, side, xml_root)).getroot()
    times: dict[str, tuple[float, float]] = {}
    for el in root.iter():
        tid = el.get(f"{NITE}id")
        if tid is None:
            continue
        try:
            s = float(el.get(f"{NITE}start"))
            e = float(el.get(f"{NITE}end"))
        except (TypeError, ValueError):
            continue
        times[tid] = (s, e)
    return times


def load_terminal_words(
    conv: int, side: str, xml_root: Path = NXT_XML
) -> list[tuple[str, str, float, float]]:
    """Timed word terminals in document order: (id, orth, start, end).

    Only ``<word>`` elements with numeric times — punctuation/silence/traces
    are excluded. The Treebank-style ``orth`` differs from ms98 tokenization
    (contractions split, case preserved); lowercase and reconcile at use site.
    """
    root = ET.parse(_side_file("terminals", conv, side, xml_root)).getroot()
    words: list[tuple[str, str, float, float]] = []
    for el in root.iter("word"):
        tid = el.get(f"{NITE}id")
        orth = el.get("orth")
        if tid is None or orth is None:
            continue
        try:
            s = float(el.get(f"{NITE}start"))
            e = float(el.get(f"{NITE}end"))
        except (TypeError, ValueError):
            continue
        words.append((tid, orth, s, e))
    return words


@dataclass(frozen=True)
class DialAct:
    conv: int
    side: str
    da_id: str
    nite_type: str
    swbd_type: str
    bases: tuple[str, ...]
    decorations: frozenset[str]
    start: float | None  # None when no child terminal carries times
    end: float | None
    n_terminals: int


def load_dialacts(conv: int, side: str, xml_root: Path = NXT_XML) -> list[DialAct]:
    """Gold dialog acts for one side, spans resolved through the terminals layer."""
    times = load_terminal_times(conv, side, xml_root)
    root = ET.parse(_side_file("dialAct", conv, side, xml_root)).getroot()
    acts: list[DialAct] = []
    for da in root.iter("da"):
        swbd = da.get("swbdType", "")
        bases, decos = parse_tag(swbd)
        ids = _href_ids(da)
        timed = [times[t] for t in ids if t in times]
        acts.append(
            DialAct(
                conv=conv, side=side,
                da_id=da.get(f"{NITE}id", ""),
                nite_type=da.get("niteType", ""),
                swbd_type=swbd, bases=bases, decorations=decos,
                start=min(t[0] for t in timed) if timed else None,
                end=max(t[1] for t in timed) if timed else None,
                n_terminals=len(ids),
            )
        )
    return acts


def load_disfluencies(conv: int, side: str, xml_root: Path = NXT_XML) -> list[dict]:
    """Disfluency groups: reparandum/repair terminal ids + their spans."""
    times = load_terminal_times(conv, side, xml_root)
    root = ET.parse(_side_file("disfluency", conv, side, xml_root)).getroot()
    out: list[dict] = []
    for disf in root.iter("disfluency"):
        rec: dict = {"disf_id": disf.get(f"{NITE}id", "")}
        for role in ("reparandum", "repair"):
            el = disf.find(role)
            ids = _href_ids(el) if el is not None else []
            timed = [times[t] for t in ids if t in times]
            rec[f"{role}_ids"] = ids
            rec[f"{role}_span"] = (
                (min(t[0] for t in timed), max(t[1] for t in timed)) if timed else None
            )
        out.append(rec)
    return out


def align_to_utterances(
    gold_spans: list[tuple[float, float]],
    utt_spans: list[tuple[float, float]],
    min_frac: float = 0.5,
) -> list[int | None]:
    """Best-overlap utterance index per gold span, or None below threshold.

    A gold span matches the utterance with maximal temporal overlap, accepted
    only if the overlap covers ≥ ``min_frac`` of the SHORTER of the two spans
    (the audit D3 rule). Same-side spans on the shared conversation clock;
    input order is free. O((n+m)·window) via a sorted sweep with a
    running-max-end early stop.
    """
    order = sorted(range(len(utt_spans)), key=lambda i: utt_spans[i][0])
    starts = [utt_spans[i][0] for i in order]
    ends = [utt_spans[i][1] for i in order]
    runmax: list[float] = []
    cur = float("-inf")
    for e in ends:
        cur = max(cur, e)
        runmax.append(cur)

    out: list[int | None] = []
    for gs, ge in gold_spans:
        best_i, best_ov = None, 0.0
        for k in range(bisect_left(starts, ge) - 1, -1, -1):
            if runmax[k] <= gs:
                break  # nothing earlier can reach into the gold span
            ov = min(ge, ends[k]) - max(gs, starts[k])
            if ov > best_ov:
                best_ov, best_i = ov, order[k]
        if best_i is None:
            out.append(None)
            continue
        shorter = min(ge - gs, utt_spans[best_i][1] - utt_spans[best_i][0])
        out.append(best_i if (shorter > 0 and best_ov >= min_frac * shorter) else None)
    return out
