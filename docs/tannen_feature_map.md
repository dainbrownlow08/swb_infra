# Switchboard Feature Extraction ↔ Tannen's Conversational Style

A reference mapping our current acoustic + linguistic feature columns to
Deborah Tannen's *Conversational Style: Analyzing Talk Among Friends*
(Oxford, 2005). PDF page references match `tannen_features.txt`
(PDF page = printed book page + 21).

---

## Part 1 — What we currently extract

Source columns from `legacy/aligned_acoustic_linguistic_v2.csv`. Extractor
code lives in `src/swb_extract/features/` (the rewritten version) with the
original references in `legacy/Conversational-Styles/src/feature_extractors/`.

| # | Column | What it computes | Source code | Tannen dimension served |
|---|--------|------------------|-------------|-------------------------|
| 1 | **Gender** | LDC `caller_tab.csv` sex field | `features/demographics.py` | (covariate, not a Tannen feature) |
| 2 | **Region** | LDC dialect_area | `features/demographics.py` | (covariate) — Tannen's NY/CA/UK speaker contrast hinges on this |
| 3 | **Year Born / Generation / Decade** | LDC birth_year + bucketed | `features/demographics.py` | (covariate) |
| 4 | **Education** | LDC education code (0–3, 9) | `features/demographics.py` | (covariate) |
| 5 | **pitch mean** | mean F0 (Hz) over voiced frames, librosa.pyin (50–400 Hz) | `features/pitch.py` | Ch.7 dim 2b (pitch); Ch.2 "marked pitch shifts"; Expressive Phonology — PDF p. 128 |
| 6 | **pitch std** | F0 standard deviation | `features/pitch.py` | Ch.7 dim 2b (marked pitch *shifts*) — PDF p. 202 |
| 7 | **pitch range** | F0 max − min | `features/pitch.py` | Ch.7 dim 2b; "marked pitch and amplitude shifts" — PDF p. 62 |
| 8 | **loudness mean** | mean RMS energy from STFT | `features/loudness.py` | Ch.7 dim 2a (loudness) — PDF p. 202 |
| 9 | **loudness std** | RMS std | `features/loudness.py` | Ch.7 dim 2a; "increased amplitude" floor-getting device — PDF p. 202 |
| 10 | **loudness range** | RMS max − min | `features/loudness.py` | Ch.7 dim 2a; "marked amplitude shifts" — PDF p. 62 |
| 11 | **Turn Gap** | start_time − previous utt end_time (sec) | `feature_extractors/turn_gap_seconds.py` | Ch.7 dim 5b (timing of contribution rel. to previous); "Avoiding interturn pauses" — PDF p. 61, 202 |
| 12 | **Filler Word Rate** | count of {um, uh, like, you know, i mean, so, well, i guess, basically, er} / total tokens | `features/filler_word_rate.py` | Loosely: Ch.7 dim 2c (pauses — fillers as filled pauses); also a counter-marker of high-considerateness hesitancy. Tannen does not name fillers as a primary feature. |
| 13 | **Pronoun Rate** | PRON-tagged tokens / total tokens (spaCy) | `features/pronoun_rate.py` | Weak proxy for Ch.7 dim 1 (relative personal focus of topic) and Ch.4 "Mutual Revelation" (PDF p. 122) — first/second person density |
| 14 | **Repetition Rate** | unique tokens appearing ≥2 times within utterance / total tokens | `features/repetition_rate.py` | Ch.7 dim 5d (floor-getting via repetition of words) — PDF p. 202 |
| 15 | **syllable_rate** | textstat syllables / utterance duration | `features/syllable_rate.py` | Ch.7 dim 5c (rate of speech); Ch.2 "Faster rate of speech" — PDF p. 61, 202 |
| 16 | **token_count** | tokens per utterance | `feature_extractors/token_count_and_rate.py` | (volume covariate; underlies "tells more / talks more") |
| 17 | **word_rate** | words / duration (sec) | `feature_extractors/word_rate_v2.py` | Ch.7 dim 5c — duplicates syllable_rate at coarser grain |
| 18 | **repetitions_in_current_utterance** | lemma-pair matches within utt | `feature_extractors/cross_utterance_repeats.py` | Ch.7 dim 6a (repetition to add to one's own line); also speech disfluency proxy |
| 19 | **repetitions_in_previous_utterance** | lemma matches between current and previous utt | `feature_extractors/cross_utterance_repeats.py` | Ch.7 dim 6 (repetition to incorporate other's offer); Ch.4 example "rightrightrightright" (PDF p. 117) |

Also present in the codebase but not in v2 CSV: `filler_word_per_second`,
`pronoun_per_second`, `repetition_per_second` (denominator = duration),
and a `politeness_score` extractor that wraps ConvoKit's PolitenessStrategies
classifier trained on the Wikipedia politeness corpus.

---

## Part 2 — Coverage of Tannen's framework

Mapping the 9 dimensions and 5 specific devices in the Ch.7 summary
(PDF p. 202–203) to what we have now. ✓ = at least partially covered;
~ = weak proxy only; ✗ = not extracted.

### Ch.7 dimensions

| Dimension | Status | Current proxy | Gap |
|-----------|--------|---------------|-----|
| 1. Relative personal focus of topic | ~ | Pronoun Rate (1st/2nd person density) | No topic classifier; no "talking-about-self" detector |
| 2a. Loudness | ✓ | loudness mean/std/range | None |
| 2b. Pitch | ✓ | pitch mean/std/range | None |
| 2c. Pauses | ~ | Turn Gap covers between-turn; filler rate covers filled pauses | **Within-turn silent pauses unmeasured** despite word-level alignment available |
| 2d. Voice quality and tone | ✗ | — | No jitter, shimmer, HNR, spectral tilt, breathiness, creak |
| 3a. Quickness of response | ✓ | Turn Gap | Should distinguish *positive* (very short / latched) from *long* gap |
| 3b. Paralinguistics for enthusiasm | ~ | pitch/loudness ranges | No "marked shift" detector — current stats are summary moments |
| 3c. Free offer of related material | ✗ | — | Requires discourse modelling: spontaneous topic addition vs. response |
| 3d. Use of questions | ✗ | — | No question detection (syntactic or prosodic) |
| 4a. Echo questions as back-channel | ✗ | — | Detect when current utt ≈ previous utt restated as question |
| 4b. Information questions | ✗ | — | Wh-/yes-no question rate |
| 5a. Cooperative vs. obstructive overlap | ✗ | — | Requires aligned A+B side timing — feasible from word.text |
| 5b. Timing of contribution | ✓ | Turn Gap | Need negative gaps (overlap) preserved, latching (~0s) flagged |
| 5c. Rate of speech | ✓ | syllable_rate, word_rate | None |
| 5d. Floor-getting (amplitude, repetition) | ~ | loudness range, Repetition Rate | No turn-onset amplitude vs. baseline; no "machine-gun" repetition pattern |
| 6a. Repetition to extend another's line | ~ | repetitions_in_previous_utterance | Lemma overlap is coarse — no "completion of other's syntax" detector |
| 6b. Repetition to incorporate other's offer | ~ | same column | Same caveat |
| 7. Topic cohesion / tolerance for diffuse topics | ✗ | — | No topic-shift / coherence metric |
| 8. Tolerance for noise vs. silence | ~ | Turn Gap distribution | No conversation-level silence ratio; no overlap density |
| 9. Laughter (when, how much) | ✗ | — | `[laughter]` brackets are *currently stripped* by all our extractors — losing the signal |

### Ch.7 specific devices

| Device | Status | Notes |
|--------|--------|-------|
| Machine-gun questions | ✗ | Composite: high pitch + reduced syntax + fast pace + question intent + short turn gap. All ingredients except question detection are extractable. |
| Mutual revelation / personal statements | ✗ | Needs first-person disclosure classifier; pronoun rate is a weak floor. |
| Ethnically-marked / in-group expressions | ✗ | Could detect Yiddishisms, AAVE markers, regional lexicon via dictionary. |
| Story rounds | ✗ | Needs narrative segmentation across multi-utterance spans. |
| Ironic or humorous routines | ✗ | `[laughter]` bracket detection would be a lower bound; Cutler's irony cues (PDF p. 185) — nasalization, slow rate, exaggerated stress — partially extractable. |

---

## Part 3 — Proposed additional features

The Switchboard data we have access to:

- **Stereo per-channel audio** sliced per utterance (`utterances/`, `utterances_v2/`)
- **Word-level alignments** with start/end timestamps in
  `swb_ms98_transcriptions_cleaned/*/*/sw*-word.text`
- **Utterance-level alignments** in the parallel `*-trans.text` files
- **Caller metadata** in `tables/caller_tab.csv` and topic in `topic_tab.csv`

This gives us much richer raw material than the current column set uses.
Suggestions are grouped by Tannen feature, with brief implementation notes.

### Pacing & turn-taking (Ch.4 Overlap and Pace, PDF p. 113–122)

1. **within_utterance_pause_count** / **mean_intra_pause_sec** / **max_intra_pause_sec**
   — Word-level alignments give silence between adjacent words inside one
   utterance. Tannen's "strategic within-turn pauses" (Ch.2 feature 4d,
   PDF p. 62) is currently invisible to us.

2. **filled_pause_rate** — separate `um`/`uh` (filled pauses) from
   `like`/`you know`/`i mean` (discourse markers). Today they're lumped
   into `Filler Word Rate`. The two have *opposite* readings: filled
   pauses signal hesitation (high-considerateness); discourse markers like
   "y'know" signal involvement (PDF p. 129 Yiddish "Oy" example).

3. **latching_flag** — `Turn Gap` ≤ 50 ms (or negative). Tannen treats
   latching as the canonical involvement signal (PDF p. 119: "rapid rate
   of speech, overlap, and latching… show solidarity").

4. **overlap_duration_sec** / **overlap_count** — Requires merging A and B
   word-timed transcripts. Distinguishes *cooperative* overlap (Tannen's
   high-involvement marker) from interruptions. This is the single most
   important Tannen feature we don't compute.

5. **turn_length_sec** — already in `legacy/.../turn_length_seconds.py`.
   Not in current CSV. Underlies "talked the most" measurements (PDF p. 124).

### Paralinguistics & voice quality (Ch.7 dim 2d, PDF p. 202)

6. **jitter / shimmer / HNR** — Standard Praat-style voice quality features.
   Tannen's "marked voice quality" (PDF p. 62) and "thick quality"
   (PDF p. 136) currently has no acoustic correlate in our pipeline.

7. **spectral_tilt** / **H1-H2** — Captures breathiness and creak. The
   "breathy voice quality" Tannen describes for empathy (PDF p. 130) and
   the "creak" / vocal fry that marks turn-end can be measured directly.

8. **pitch_slope** and **pitch_excursion_per_syllable** — Tannen distinguishes
   "marked pitch shifts" (PDF p. 62) from raw range. A high range can come
   from one outlier; slope captures the contour shape (e.g. her "He's a
   gréat writer" example, PDF p. 129).

9. **loudness_onset_delta** — RMS in first 200 ms of turn vs. session mean.
   Operationalizes "increased amplitude" as a floor-getting device
   (Ch.7 dim 5d, PDF p. 202).

### Question and prosodic-form features (Ch.7 dim 4, PDF p. 202)

10. **question_flag** (syntactic) — Wh-word at start, subj-aux inversion,
    or final `?` if any pretokenized form exists. Required for both
    "machine-gun questions" and "echo questions as back-channel".

11. **rising_terminal_flag** — Last 200 ms F0 slope > threshold. Catches
    declarative-form questions ("You stayed at the Plaza?" PDF p. 113)
    that syntactic detection misses.

12. **machine_gun_question_score** — Composite z-score: (high pitch) +
    (short syntax: ≤4 words) + (short Turn Gap) + (question_flag). Exactly
    the four ingredients Tannen names at PDF p. 112.

13. **echo_question_flag** — Current utt is question AND ≥80% lemma
    overlap with previous turn's content words (e.g. Tannen p. 87 "You
    stayed at the Plaza?" echoing Chad).

### Repetition & alignment (Ch.7 dim 6, PDF p. 202)

14. **other_repetition_count** / **self_repetition_count** — Already partly
    have `repetitions_in_previous_utterance`, but this lumps *both*
    speakers together. Tannen distinguishes repeating *the other* (rapport
    — "rightrightright", PDF p. 117) from repeating *self* (insistence).

15. **collaborative_completion_flag** — Current turn syntactically/lexically
    completes the previous speaker's unfinished turn (e.g. Tannen's "would-be
    actors" example, PDF p. 71). Detect via partial-word brackets `i[t]-`
    in previous turn + immediate semantic continuation.

16. **lexical_alignment_score** — Cosine similarity of sentence embeddings
    between adjacent A and B turns over a moving window. Captures
    "thematic cohesion" (Ch.7 dim 7).

### Laughter, humor, irony (Ch.6 + Ch.7 dim 9, PDF p. 184, 202)

17. **laughter_token_count** — Add a dedicated `[laughter]` counter (and
    one for `[noise]`, `[vocalized-noise]` etc.). Tannen's dim 9 deserves
    its own column. Note the current state is *messy*, not absent: the
    legacy extractors that populated the v2 CSV do **not** strip brackets,
    so `[laughter]` is sitting in the denominator of every rate (and is
    counted as a real token by `Repetition Rate` if it appears twice in
    one utterance). The rewritten `src/swb_extract/features/` versions
    do strip — but they're not in the v2 CSV yet. Either way, no clean
    laughter count exists today.

18. **laughter_audio_event_count** — Use a laughter detector (e.g. OpenSMILE's
    Laughter detector, or Wav2Vec2 fine-tuned for nonverbal vocal events)
    to catch laughs not transcribed.

19. **irony_prosody_score** — Cutler's three cues quoted by Tannen
    (PDF p. 185): nasalization, slowed rate, exaggerated stress.
    Approximations: spectral nasal energy ratio (1–2 kHz dip),
    syllable_rate vs. speaker baseline, peak-stress-syllable F0+RMS z-score.

### Topic, content, and discourse (Ch.7 dim 1 & 7, PDF p. 202)

20. **personal_focus_score** — Ratio of (1st-person + 2nd-person pronouns +
    personal-content noun phrases) to total content words. Better proxy
    for "relative personal focus of topic" than raw pronoun rate.

21. **topic_label** — Switchboard ships `tables/topic_tab.csv` with the
    assigned topic per call. Free covariate for the Tannen "personal vs.
    impersonal topic" split (Ch.4 Personal vs. Impersonal Topics, PDF p. 90).

22. **topic_shift_count** — Per-call count of LDA / topic-embedding shifts.
    Directly indexes Tannen's "tolerance for diffuse topics" (Ch.7 dim 7).

23. **mutual_revelation_flag** — Current turn is a first-person past-tense
    personal anecdote AND previous turn was the same. ConvoKit's
    `disclosure` annotators or simple regex (`I (was|did|had|went) …`)
    works as v1.

24. **politeness_score** — Already implemented in
    `feature_extractors/politeness_score.py` (ConvoKit
    PolitenessStrategies). Plug it into the pipeline. It quantifies the
    Lakoff "Don't impose" / "Be friendly" axis Tannen builds her whole
    framework on (PDF p. 37).

### Narrative / story features (Ch.5, PDF p. 144–183)

25. **narrative_turn_flag** — Past-tense + character/event NER + ≥N tokens.
    Replicates Tannen's manual Table 5.1 narrative count (PDF p. 145).

26. **story_round_flag** — ≥3 consecutive narrative turns by ≥2 speakers
    sharing topic embedding. Tannen's story-round definition exactly
    (PDF p. 147).

27. **constructed_dialogue_count** — Count quoted-speech spans inside a
    narrative turn (regex for `said`, `says`, plus quote marks if any
    surface form exists). Tannen's "Meaning in Intonation" device
    (PDF p. 172).

28. **narrative_intonation_dramatization_score** — pitch_range +
    loudness_range *within* narrative turns vs. baseline. Operationalizes
    the "dramatized rather than lexicalized" point preference
    (Ch.2 feature 3c, PDF p. 62).

### Conversation-level / speaker-level aggregates

These are derived by groupby on the per-utterance table — cheap to add
once the per-utterance signals exist.

29. **speaker_silence_ratio** — Σ silence / Σ call duration, per speaker.
    Direct measure of "tolerance for silence" (Ch.7 dim 8).
30. **speaker_overlap_initiation_rate** — Overlaps per minute initiated by
    speaker. Cooperative-overlap signal.
31. **speaker_persistence_score** — Tannen's "two, three, and four tries"
    (PDF p. 131): count of dropped topics that the same speaker
    re-introduces within N turns.
32. **speaker_narrative_share** — % of turns that are narrative.
    Reproduces Tannen's Table 5.1 ranking (PDF p. 145).
33. **speaker_humor_rate** — % of turns containing `[laughter]` either
    before/within/after them (audience or self). Reproduces Table 6.1
    (PDF p. 185).

---

## Notes on what we're losing today

Two recurring issues worth flagging:

- **Bracket annotation handling is inconsistent and unmeasured.** The
  legacy extractors that produced `aligned_acoustic_linguistic_v2.csv`
  do **not** strip `[laughter]` / `[noise]` / `[vocalized-noise]` —
  `within_utterance_repetition.py`, `filler_word_rate.py`,
  `syllable_rate_textstat.py`, `pronoun_rate_spacy.py`, and
  `word_rate_v2.py` all pass raw transcript text to `split()` /
  textstat / spaCy. Effects: bracket tokens inflate the denominator of
  every rate; `Repetition Rate` will fire on a doubled `[noise]`;
  textstat syllabifies "laughter" inside the brackets. The rewritten
  `src/swb_extract/features/` extractors strip brackets cleanly — but
  they aren't in the production CSV yet. Net result: laughter (Tannen's
  ninth dimension, PDF p. 202) is neither cleanly counted nor cleanly
  removed. Adding a dedicated bracket-event counter pass before
  stripping fixes both problems at once.

- **Per-utterance summary statistics flatten exactly the contrasts Tannen
  cares about.** "Marked pitch shifts" (Ch.2 4b) means *how* the F0
  contour moves, not its mean or range. A speaker can have identical
  pitch_range to another while one speaks in a flat declarative and the
  other in dramatic exclamations. Sequence features (slope per syllable,
  number of pitch reset points, contour entropy) would capture what
  static moments cannot.
