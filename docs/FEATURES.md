# Feature extractor trust registry

> **Living document.** The single source of truth for which feature columns the
> trustworthy/experimental analysis may use. `src/swb_extract/registry.py` **parses
> this file**, and the loader (`load_features_table`) enforces it. To change a
> feature's trust, **move its row between sections** — that is the whole workflow.
>
> **Buckets** (the section a row lives under = its status):
> - **Trusted** — correctness confirmed; analysis may rely on it.
> - **WIP** — built and producing values, but not yet confirmed. Usable in
>   experiments; the loader prints a warning naming every WIP column it hands you.
> - **Deprecated** — do not use. Kept for paper replication only.
>
> Promote WIP → Trusted only after the extractor is confirmed (unit tests + a
> distribution sanity check + ideally the NXT-gold check, audit §4C). See
> `docs/PIPELINE.md` for the add-a-feature and recompute procedure.
>
> _Last reviewed: 2026-07-09 (NB07 Step 11 trust adjudication, submission plan T3 —
> 26 rows promoted on printed evidence; per-utterance Personal Focus Score
> deprecated). Counts: 43 Trusted · 6 WIP · 5 Deprecated. Live dashboard:
> `python3 -c "import sys;sys.path.insert(0,'src');from
> swb_extract import registry as R;print(R.summary())"`._

## Trusted

| Column | Extractor | Family | Notes |
|---|---|---|---|
| token_count | token_count.py | volume | 0% null, 1–81 tokens, med 8 |
| word_rate | word_rate.py | volume | 0% null, med 2.29/s |
| syllable_rate | syllable_rate.py | volume | 0% null, med 2.90/s |
| loudness mean | loudness.py | volume | linear RMS 0–0.12; z-score before use |
| loudness std | loudness.py | volume | 0% null |
| loudness range | loudness.py | volume | 0% null |
| Pronouns per Second | pronoun_per_second.py | volume | 0% null; per-token variant deprecated |
| Repetitions In Current Utterance | repetitions_in_current.py | volume | pair count Σ C(n,2) BY DESIGN — max 109 > max token_count 81 is the quadratic tail (boundary test pins C(15,2)=105); full-corpus recompute reconciled (NB07 Step 11); heavy right tail — consider log1p at analysis time (audit §2.8). Gold de-conflation DONE (NB07 Step 13): 24.5% of pair-repetitions attributable to gold repair material (de-conflated variant computed, 544-conv subset; extractor↔gold r=.978); ≈0 side-level correlation with gold mirror `^m` (sparse ~0.5/side) — self-repetition ≠ allo-repetition; involvement-relevant repetition = the gold mirror rate (T9) |
| Repetitions In Previous Utterance | repetitions_in_previous.py | volume | cross-utterance pair matches vs chronological predecessor; 0.9% null = conversation-initial, verified ≤1/conversation (NB07 Step 11); same tail caveat; gold de-conflation (Step 13): 14.2% repair-attributable (time-ordered-predecessor approximation) |
| Repetitions per Second | repetition_per_second.py | volume | numerator = repetition_rate count (unique words reaching ≥2), NOT the pair count — by design per docstring; shares tokenize with the repetition family (NB07 Step 11) |
| Filler Words per Second | filler_word_per_second.py | volume | allowlist verified vs docstring (um/uh/er + so/well/like/you know/i mean/i guess/basically, phrase-aware); filled-pause vs discourse-marker UNSPLIT — opposite theoretical signs may cancel (audit §4E-c) |
| FTO Sec | fto.py | interactional | floor-transfer offset; 64.5% null BY DESIGN (floor transfers only); med +0.14s, matches Heldner & Edlund (audit §3.1) |
| Onset Gap Sec | fto.py | interactional | 25.4% null = backchannels + conversation-initial turns, decomposed (NB07 Step 11); rides tests/test_fto.py |
| Turn Initial Flag | fto.py | interactional | FTO defined ⇒ flag=1 verified corpus-wide (NB07 Step 11); rides tests/test_fto.py |
| Backchannel Flag | fto.py | interactional | lexical allowlist — corpus-wide agreement with the notebook is_bc verified (NB07 Step 11); allowlist measured vs NXT gold (Step 12): P .842 / R .917 on primary {b}, F1 .878 |
| Interjection Flag | fto.py | interactional | contained other-speaker utterance (no floor transfer); rides tests/test_fto.py; lexical-allowlist heuristic — NXT gold P/R pending (§4C12) |
| Latching Flag | latching_flag.py | interactional | Latching=1 ⇒ raw FTO ∈ [0, 0.2] s verified corpus-wide at 0 violations AFTER Step 11 caught a stale pre-Jun-26-FTO vintage (1,333/214,204 rows corrected on re-extraction 2026-07-09); 17.9% of defined transfers latched; LATCH_MAX_SEC=0.2 documented + sweepable |
| Overlap Duration Sec | overlap.py | interactional | word-level intervals; cooperative/obstructive split → overlap_split extractor (§4E-a, submission plan T6) |
| Overlap Count | overlap.py | interactional | 0% null among placeable; verified vs negative-FTO coherence (NB07 Step 11) |
| Overlap Onset Flag | overlap.py | interactional | onset-in-overlap coheres with FTO<0 at transfers (NB07 Step 11) |
| Within Pause Total Sec | within_utterance_pauses.py | interactional | sums ALL positive inter-word gaps (Count only ≥0.25 s — so Count=0 does NOT imply Total=0); invariants verified corpus-wide (NB07 Step 11); zero-inflated |
| Within Pause Count | within_utterance_pauses.py | interactional | gaps ≥ PAUSE_MIN_SEC=0.25; Count≥1 ⇔ Max≥0.25 verified (NB07 Step 11) |
| Within Pause Rate | within_utterance_pauses.py | interactional | Total/span; ≤1 verified corpus-wide (NB07 Step 11) |
| Max Within Pause Sec | within_utterance_pauses.py | interactional | ≤ Total verified corpus-wide (NB07 Step 11) |
| Rising Terminal Flag | rising_terminal.py | interactional | 30.3% null MNAR, voicing/length-dependent — null rate RISES with utterance length (21%→35% for ≤2→≥6 tokens; NB07 Step 11), contra the docstring's short-tails guess; do NOT fillna(0) — treat null as indeterminate |
| Terminal F0 Slope | rising_terminal.py | interactional | clip at ±1000 Hz/s before use (bound chosen + recorded, NB07 Step 11) |
| Laughter Count | laughter.py | interactional | full-corpus bracket reconciliation vs Transcript exact (NB07 Step 11); RATE-normalize at analysis — raw counts are a talkativeness proxy (Jun-19 audit) |
| Laughed Word Count | laughter.py | interactional | reconciled corpus-wide (NB07 Step 11) |
| Noise Count | laughter.py | interactional | counted before stripping (audit §3.5) |
| Vocalized Noise Count | laughter.py | interactional | counted before stripping (audit §3.5) |
| Other Bracket Count | laughter.py | interactional | 255 corpus-wide — negligible |
| Personal Hits | personal_focus_score.py | tannen | raw ingredient — pool-then-ratio at unit level ONLY (coverage + non-degeneracy evidence NB07 Step 11); Empath category mapping unvalidated — carry the §4C12 caveat in any analysis |
| Impersonal Hits | personal_focus_score.py | tannen | raw ingredient for pooling (see Personal Hits note) |
| Analyzed Tokens | personal_focus_score.py | tannen | density basis for pooling |
| pitch mean | pitch.py | prosody | 8% null (unvoiced); 50–400 Hz clamp; static moment only |
| pitch std | pitch.py | prosody | 8% null |
| pitch range | pitch.py | prosody | 8% null |
| Gender | demographics.py | meta | corpus covariate (114k F / 99k M) |
| Region | demographics.py | meta | corpus covariate (9 regions) |
| Year Born | demographics.py | meta | corpus covariate (1924–1975) |
| Generation | demographics.py | meta | derived from Year Born |
| Decade | demographics.py | meta | derived from Year Born |
| Education | demographics.py | meta | corpus covariate (0–9) |

## WIP

| Column | Extractor | Family | Notes |
|---|---|---|---|
| Question Flag | question_flags.py | interactional | VALIDATED vs NXT gold 2026-07-09 (NB07 Step 12, 52,890 labelled utts): precision .553 / recall .236 (recall by type: syntactic 29%, declarative 6%, tag 3%); gold q-rate 7.83% vs flag 3.34% — the audit's rate-gap explained; EXCLUDED from analyses per the pre-registered 0.8 bar; Tier-3 marker-skip fix declined on evidence (+716 gain vs +471 new FPs); classifier route = T7 (§4C12) |
| Echo Question Flag | question_flags.py | interactional | adjudicated 2026-07-09 vs gold `bh` (backchannel-in-question-form): precision .209 / recall .025 — construct mismatch; keep out of analyses; gold `bh` itself is the panel variable (T9) |
| Machine Gun Question Score | machine_gun_question.py | interactional | 49.7% null — decomposition pending (audit §3.4 / C1 Tier 4); ingredients gated on T4; FTO-derived → same stale-vintage risk Step 11 caught in latching — RE-EXTRACT before first use |
| Machine Gun Question Flag | machine_gun_question.py | interactional | 49.7% null |
| Mutual Revelation Flag | mutual_revelation_flag.py | tannen | EXCLUDED from the paper — spot-checked precision ~30–40% (Jun-19 audit); no gold rescue exists; keep out until hand-labeled |
| Topic Label | topic_label.py | meta | assigned SWBD topic; join-validity unchecked, never yet joined (audit §2.6/§4D16 — out of submission scope) |

## Deprecated (replication only)

| Column | Extractor | Family | Notes |
|---|---|---|---|
| Turn Gap | turn_gap.py | interactional | legacy gap vs chronological predecessor incl. backchannels; 60% negative; superseded by FTO Sec |
| Pronoun Rate | pronoun_rate.py | volume | per-token variant; using Pronouns per Second instead |
| Repetition Rate | repetition_rate.py | volume | per-token variant; using Repetitions per Second instead |
| Filler Word Rate | filler_word_rate.py | volume | per-token variant; using Filler Words per Second instead |
| Personal Focus Score | personal_focus_score.py | tannen | DEGENERATE per-utterance ratio — 71.5% null, saturates 0/1 (Jun-19 audit 2/10); pool the raw hit columns instead (NB07 Step 11; audit §3.3) |
