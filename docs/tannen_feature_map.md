# Tannen's Conversational Style ↔ the Thomas et al. (2018) feature set

**Scope (decision 2026-08-20).** This map covers the **eleven variables of Thomas, Czerwinski,
McDuff, Craswell & Mark (2018), "Style and Alignment in Information-Seeking Conversation"
(CHIIR '18, doi:10.1145/3176349.3176388)** as rebuilt in `src/swb_extract/features/thomas2018/`
— and nothing else. The in-house extractors' coverage (every earlier version of this file,
including the "row 12" / "item #17" numbering some in-house docstrings cite) is archived
verbatim at `docs/archive/tannen_feature_map_inhouse_2026-08-20.md`; expanding the map to
Thomas + in-house is later work. Trust state lives in `docs/FEATURES.md` (none of the eleven
is registered yet — built, not extracted; status in `docs/AUDIT.md` §4E-h).

**Sources and labels.** Tannen = *Conversational Style: Analyzing Talk Among Friends* (Oxford,
2005); page references are PDF pages (= printed page + 21), matching the quotes collected in
`tannen_features.txt`. Thomas et al. built their Table 1 on Tannen's 1987 summary
("Conversational style", in Dechert & Raupach, eds., *Psycholinguistic Models of Production*),
which is the same checklist as the 2005 book's Ch.2 "features of high-involvement style"
(PDF pp. 61–62) with the wording differences listed in Part 3. Every claim below carries one
of three labels:

- **[T1]** — Thomas et al.'s own assignment of a variable to a Tannen characteristic (their Table 1).
- **[PDF p. N]** — Tannen's 2005 text, by PDF page. The only source for a HI/HC *direction*:
  an item is signed only by appearing on the Ch.2 high-involvement list (PDF pp. 61–62);
  the Ch.7 nine-dimension summary (PDF pp. 202–203) is unsigned.
- **[ours]** — a bridge, caveat, or corpus-mapping decision of this project; not a claim of
  either paper.

**Signs.** "MISC loading" is the variable's loading on Thomas et al.'s "involvement" component
(their Table 2: a PCA over the MISC participant-task observations, first component 29% of
variance; the involvement score is the z-scored variables summed by these loadings). It is an
empirical result on one small corpus of information-seeking dialogue (22 pairs), **not a
Tannen prediction**; the paper itself reports the one reversal (olap, −0.01, "practically
zero"). Thomas et al. summarise the poles by quoting Tannen 1987 — high involvement: "When in
doubt, talk. Ask questions. Talk fast, loud, soon. Overlap. Show enthusiasm. Prefer personal
topics"; high consideration: "Allow longer pauses. Hesitate. Don't impose one's topics, ideas,
personal information. Use moderate paralinguistic effects." (their §2.1; label **[T§2.1]**).

---

## Part 1 — The eleven variables

Unit: the paper's unit is participant × task; on Switchboard that is one conversation side.
Each module writes per-utterance ingredients and `aggregate()` pools them as the paper does
(ratio of sums / pooled variance / mean — never a mean of per-utterance ratios);
`swb-extract thomas2018-side` writes the side table.

| # | Variable | Paper definition (§3.2) | Here (`thomas2018/`) | Tannen anchor | Book direction (Ch.2 HI list) | MISC loading |
|---|---|---|---|---|---|---|
| 1 | **ppron** | rate of 1st/2nd-person pronouns, not 3rd ("our own list") | Σ pronouns / Σ words — the 16-form closed list; per word, the LIWC convention [ours] | Ch.7 dim 1 "relative personal focus of topic" [PDF p. 202]; Ch.2 1a "Prefer personal topics" [T1]; Ch.4 personal vs impersonal topics [PDF pp. 95, 102–103]. Caveat: pronouns measure the *framing* of talk (self/other reference), not its topic — "I think the government should …" is a personal frame on an impersonal topic [ours]. Thomas et al. expected a weak signal because MISC topics were assigned; Switchboard topics are assigned too [ours] | HI + | +0.13 |
| 2 | **wps** | words / total duration of utterances (a micro-average) | Σ`wpu` / Σ`wps sec`, word-tight utterance spans [ours] | Ch.7 dim 5c "rate of speech" [PDF p. 202]; Ch.2 2a "Faster rate of speech" [PDF p. 61] [T1 "Faster rate"]; "rapid rate of speech, overlap, and latching … show solidarity" [PDF p. 119] | HI + | +0.39 |
| 3 | **wpu** | mean words per utterance | mean of `wpu` | [T1] files it under "Faster rate" (Pace). The 2005 checklist has no item for utterance length as such; the nearest anchor is floor share — who "talked the most" [PDF p. 124] — so here it is a volume measure, a bridge [ours]. The paper's largest loading; in this corpus utterance length is also the backchannel/segmentation axis (AUDIT §1) [ours] | none in Ch.2; only "When in doubt, talk" [T§2.1] | +0.45 |
| 4 | **wpp** | words / number of between-own pauses, "approximately the length of each spoken phrase" | Σ`wpu` / Σ`wpp pauses`; pauses = unvoiced runs between voiced frames inside the utterance | [T1] "Faster rate". Ch.7 dim 2c pauses, "absolute use and use of marked shifts" — no direction attached [PDF p. 202]. Because every unvoiced run counts (as in the paper), wpp here is nearer words-per-voiced-stretch than phrase length [ours] | none stated | +0.44 |
| 5 | **boplen** | mean length of between-own pauses ("no speech signal … during a single utterance") | Σ`boplen sec` / Σ`boplen n`, the same pauses | Ch.7 dim 2c pauses, within-turn half [PDF p. 202]. [T1] "Pauses avoided". The book does not sign within-turn silence that way: Ch.2 2c "Avoiding interturn pauses" is *between* turns [PDF p. 61], while 4d "Strategic within-turn pauses" sits on the *involvement* list [PDF pp. 61–62]; "Hesitate" is the consideration pole in the 1987 summary [T§2.1]. So the [T1] label is their reading, and the MISC loading is the only empirical sign [ours; AUDIT §3 6h] | ambiguous — 4d HI + ("strategic"), hesitation HC | −0.10 |
| 6 | **poplen** | mean post-other pause: partner's end → own next start, non-alternation allowed | mean of `poplen`; word-tight; blank when the utterance starts in overlap or after own talk [ours] | Ch.7 dim 3a "quickness of response" and 5b "timing of contribution, relative to previous contribution" [PDF p. 202]; Ch.2 2b "Faster turn taking", 2c "Avoiding interturn pauses (silence shows lack of rapport)" [PDF p. 61] [T1 "Pauses avoided", "Faster rate of turntaking"]; mismatched pause expectations as a dominance mechanism [PDF p. 209 (printed 188); AUDIT §3 6h]. Every transcript line counts, backchannels included, as in the paper [ours] | HI − (shorter) | −0.27 |
| 7 | **pv** | variance of F0 over frames with a speech signal, across the whole recording | pooled variance of voiced-frame F0, Hz² | Ch.7 dim 2b pitch [PDF p. 202]; Ch.2 4b "Marked pitch and amplitude shifts" [PDF p. 62] [T1 "Pitch shifts"]; "marked pitch shifts and exaggerated stress … expressiveness and empathy" [PDF p. 130]. Caveat: a variance is a static moment — equal variances can hide entirely different *shift* patterns — and Hz² grows with mean F0, so it is partly a pitch-level measure [ours] | HI + (read as "marked shifts") | +0.09 |
| 8 | **lv** | loudness variance, "measured the same way" | pooled variance of voiced-frame RMS (OpenSMILE loudness is not a dependency) [ours] | Ch.7 dim 2a loudness [PDF p. 202]; Ch.2 4b amplitude shifts [PDF p. 62] [T1 "Loudness shifts"]; dim 5d floor-getting by "increased amplitude" [PDF p. 202] — lv is variation, not onset amplitude, so 5d is indirect [ours]. Telephone-line gain confounds between-caller level [ours] | HI + (read as "marked shifts") | +0.21 |
| 9 | **olap** | proportion of utterances that begin while the partner is still talking | mean of `olap`, word-tight | Ch.7 dim 5a "cooperative versus obstructive overlap" [PDF p. 202]; Ch.2 2d "Cooperative overlap" [T1], 2e "Participatory listenership" [PDF pp. 61–62]; overlap-as-enthusiasm [PDF p. 98]. olap does not separate cooperative from obstructive — the paper says so ("need not be an interruption … commonly … 'uh-huh'") — so it covers 5a's *rate*, not its contrast [ours]. In Switchboard the side-level olap median is .36 (IQR .24–.49, corpus-wide 2026-08-22), the overlapping starts being mostly listener backchannel lines [ours] | HI + | −0.01 (the paper's noted reversal) |
| 10 | **rept** | mean number of terms repeated from the same person's previous utterance (stopwords + um/uh/uh-huh removed, stemmed) | mean of `rept`; frozen NLTK stopwords + Porter; a term is a type [ours] | [T1 "Persistence"] = Ch.2 1d "Persist (if a new topic is not immediately picked up, reintroduce it, repeatedly if necessary)" [PDF pp. 61–62]; "persist with contributions for two, three, and four tries" [PDF p. 131]; persistence ↔ "tolerance for … diffuse talk" (dim 7) [PDF p. 140]. **Not** Ch.7 dim 6 (repetition of *another's* words, PDF p. 202): by definition rept is self-repetition across own consecutive lines [ours] | HI + (via persist) | +0.39 |
| 11 | **repu** | fraction of utterances with ≥ 1 such repeated term | mean of `repu` | as rept — the incidence view of re-statement [T1 "Persistence"] | HI + (via persist) | +0.39 |

Reading the last two columns together: where the 2005 checklist gives a direction, the MISC
sign agrees for ppron, wps, poplen, pv, lv, rept and repu; disagrees for olap (which the
paper discounts); and boplen is the case where the book's own direction is unsettled. wpu
and wpp carry the two largest loadings with no checklist direction behind them — they are
the paper's operationalization, not Tannen's.

---

## Part 2 — Coverage of Tannen's framework by the eleven

✓ = covered at least partially · ~ = weak or indirect proxy only · ✗ = nothing in the set.

### Ch.7 dimensions [PDF pp. 202–203]

| Dimension | Status | Thomas-set proxy | What the eleven cannot see |
|---|---|---|---|
| 1. Relative personal focus of topic | ~ | ppron [T1] | the topic itself (framing ≠ topic) |
| 2a. Loudness | ~ | lv — variation only | absolute level; marked *shifts* as dynamics |
| 2b. Pitch | ~ | pv — variation only | level; contour/shift dynamics; a level-free (semitone) scale |
| 2c. Pauses | ✓ | boplen (within-turn), poplen (between-turn), wpp (pause density) | the "marked shifts" half; strategic vs hesitant |
| 2d. Voice quality and tone | ✗ | — | jitter / shimmer / HNR, breathiness, creak |
| 3a. Quickness of response | ✓ | poplen | — |
| 3b. Paralinguistics for enthusiasm | ~ | pv, lv | any shift / onset detector |
| 3c. Free offer of related material | ✗ | — | discourse modelling |
| 3d. Use of questions | ✗ | — | question detection |
| 4a. Echo questions as back-channel | ✗ | — | — |
| 4b. Information questions | ✗ | — | — |
| 5a. Cooperative vs obstructive overlap | ~ | olap — rate, undifferentiated | the cooperative/obstructive contrast itself |
| 5b. Timing relative to previous contribution | ✓ | poplen | latching as a category; backchannel-aware floor transfers |
| 5c. Rate of speech | ✓ | wps (with wpp, wpu as the paper's Rate group) | syllable-level rate |
| 5d. Floor-getting (amplitude, repetition) | ~ | lv, rept — both indirect | onset amplitude vs baseline; repetition *of the other* |
| 6a/6b. Repetition of another's statement / offer | ✗ | — (rept/repu are self-repetition) | allo-repetition |
| 7. Topic cohesion / tolerance for diffuse topics | ~ | rept, repu, via persistence [PDF p. 140] | topic-shift / coherence measure |
| 8. Tolerance for noise vs silence | ~ | poplen + boplen (silence side), olap (noise side) | conversation-level silence ratio; overlap density |
| 9. Laughter | ✗ | — | laughter counts |

Tally over the 19 rows: **✓ 4** (2c, 3a, 5b, 5c) · **~ 8** (1, 2a, 2b, 3b, 5a, 5d, 7, 8) ·
**✗ 7** (2d, 3c, 3d, 4a, 4b, 6, 9). Coverage is concentrated in pacing (dim 5) and pauses;
questions, laughter, voice quality, repetition of the other, and narrative are out of the
set's reach — which is what the paper itself says ("do not address genre … not 'marked voice
quality' or 'strategic pauses'").

### Ch.7 specific devices [PDF p. 203]

| Device | Status | Notes |
|---|---|---|
| Machine-gun questions | ✗ | The set has the pace ingredients (wps, poplen, pv) but no question intent; Tannen names high pitch, reduced syntax, fast rate, directness, and the pace of firing [PDF p. 112] |
| Mutual revelation / personal statements | ✗ | ppron is at most a weak floor for "personal statement" [PDF p. 122]; no disclosure detection |
| Ethnically marked / in-group expressions | ✗ | — |
| Story rounds | ✗ | genre is outside the set by design (Table 1: "Genre — —") |
| Ironic or humorous routines | ✗ | no laughter or prosodic-irony cue in the set |

### Ch.2 "features of high-involvement style" — the list Table 1 is built on [PDF pp. 61–62]

| Tannen feature (2005 wording) | Thomas et al. Table 1 row | Variable(s) |
|---|---|---|
| 1a Prefer personal topics | Prefer personal topics | ppron [T1] |
| 1b Shift topics abruptly | Shift topics abruptly | — [T1: —] |
| 1c Introduce topics without hesitation | Introduce topics without hesitance | — [T1: —] |
| 1d Persist (reintroduce a topic, repeatedly if necessary) | Persistence | rept, repu [T1] |
| 2a Faster rate of speech | Faster rate | wps, wpp, wpu [T1] |
| 2b Faster turn taking | Faster rate of turntaking | poplen [T1] |
| 2c Avoiding interturn pauses | Pauses avoided | poplen, boplen [T1] — boplen is *within*-turn; see Part 1 row 5 |
| 2d Cooperative overlap | Cooperative overlap | olap [T1] |
| 2e Participatory listenership | *(not in Table 1)* | — ; olap's "uh-huh" content is an accidental proxy [ours] |
| 3a Tell more stories | Tell more stories | — [T1: —] |
| 3b Tell stories in rounds | Tell stories in rounds | — [T1: —] |
| 3c Prefer internal evaluation (point dramatized, not lexicalized) | Point of stories is emotion of teller | — [T1: —] |
| 4a Expressive phonology | *(not in Table 1)* | — ; pv/lv at most [ours] |
| 4b Marked pitch and amplitude shifts | Pitch shifts; Loudness shifts | pv, lv [T1] |
| 4c Marked voice quality | Marked voice quality | — [T1: —] |
| 4d Strategic within-turn pauses | Strategic pauses | — [T1: —]; boplen measures within-turn pausing, not its strategic use, and the paper explicitly does not claim it [ours] |

---

## Part 3 — Notes for the later expansion (Thomas + in-house)

- **Table 1 vs the 2005 checklist.** Thomas et al. worked from Tannen 1987: "Participatory
  listenership" (2e) and "Expressive phonology" (4a) are absent from their table; 3c "Prefer
  internal evaluation" appears as "Point of stories is emotion of teller"; 1c "hesitation" as
  "hesitance". None of this changes a mapping above, but a citation should name which text
  a wording comes from.
- **Ch.7 is broader than Ch.2.** The nine-dimension summary adds questions (3d, 4), repetition
  of another's words (6), topic cohesion (7), noise vs silence (8), laughter (9) and voice
  quality (2d); the eleven reach none of 2d, 4, 6, 9.
- **Unit.** The paper's unit is participant × task (≈ conversation side); NB08's unit is the
  caller across ~9 calls. `side_table.aggregate_group` pools either grouping from
  canonical-table rows.
- **Where the in-house set already reaches** (questions — excluded on gold; laughter; the
  cooperative/obstructive split; FTO and latching; the Within-Pause family; personal-focus
  hits; rising terminal): the archived map and `docs/FEATURES.md`. Merging the two coverages
  is the next revision of this file, not this one.
