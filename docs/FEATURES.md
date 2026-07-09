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
> _Last reviewed: 2026-06-29 (initial buckets set). Counts: 17 Trusted · 33 WIP · 4
> Deprecated. Live dashboard: `python3 -c "import sys;sys.path.insert(0,'src');from
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
| FTO Sec | fto.py | interactional | floor-transfer offset; 64.5% null BY DESIGN (floor transfers only); med +0.14s, matches Heldner & Edlund (audit §3.1) |
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
| Repetitions In Current Utterance | repetitions_in_current.py | volume | SUSPECT max 109 > max token_count 81 — verify counting unit |
| Repetitions In Previous Utterance | repetitions_in_previous.py | volume | SUSPECT max 100; 0.9% null |
| Repetitions per Second | repetition_per_second.py | volume | kept per-second; shares repetition logic — confirm vs the count bug |
| Filler Words per Second | filler_word_per_second.py | volume | filled-pause vs discourse-marker not split (audit map #2) |
| Onset Gap Sec | fto.py | interactional | 25.4% null; FTO companion |
| Turn Initial Flag | fto.py | interactional | FTO helper classification |
| Backchannel Flag | fto.py | interactional | FTO helper classification |
| Interjection Flag | fto.py | interactional | FTO helper classification |
| Latching Flag | latching_flag.py | interactional | 64.4% null (FTO-based); 17.8% latched among defined (was ~1% when broken) |
| Overlap Duration Sec | overlap.py | interactional | cooperative/obstructive split deferred (audit §4E-a) |
| Overlap Count | overlap.py | interactional | 0% null |
| Overlap Onset Flag | overlap.py | interactional | 0% null |
| Within Pause Total Sec | within_utterance_pauses.py | interactional | zero-inflated (most utts none) |
| Within Pause Count | within_utterance_pauses.py | interactional | 0% null |
| Within Pause Rate | within_utterance_pauses.py | interactional | 0% null |
| Max Within Pause Sec | within_utterance_pauses.py | interactional | 0% null |
| Question Flag | question_flags.py | interactional | rate 3.26% ~half SwDA — validate vs NXT gold (audit §4C12) |
| Echo Question Flag | question_flags.py | interactional | 0% null |
| Rising Terminal Flag | rising_terminal.py | interactional | 30.3% null (down from 68.5% — anchor fix worked); do NOT fillna(0) |
| Terminal F0 Slope | rising_terminal.py | interactional | ±2000 Hz/s outliers — clip before use |
| Machine Gun Question Score | machine_gun_question.py | interactional | 49.7% null — pitch gate may be too aggressive; per-side baseline (audit §3.4) |
| Machine Gun Question Flag | machine_gun_question.py | interactional | 49.7% null |
| Laughter Count | laughter.py | interactional | NEW (never analyzed); counted before stripping (audit §3.5) |
| Laughed Word Count | laughter.py | interactional | 0% null |
| Noise Count | laughter.py | interactional | 0% null |
| Vocalized Noise Count | laughter.py | interactional | 0% null |
| Other Bracket Count | laughter.py | interactional | 0% null |
| Personal Hits | personal_focus_score.py | tannen | raw ingredient — pool at speaker level |
| Impersonal Hits | personal_focus_score.py | tannen | raw ingredient for pooling |
| Analyzed Tokens | personal_focus_score.py | tannen | density basis for pooling |
| Personal Focus Score | personal_focus_score.py | tannen | DEGENERATE 71.5% null, saturates 0/1 — pool from hits instead (audit §3.3) |
| Mutual Revelation Flag | mutual_revelation_flag.py | tannen | 0.9% null |
| Topic Label | topic_label.py | meta | assigned SWBD topic; never yet joined (audit §2.6); confound for region/gender |

## Deprecated (replication only)

| Column | Extractor | Family | Notes |
|---|---|---|---|
| Turn Gap | turn_gap.py | interactional | legacy gap vs chronological predecessor incl. backchannels; 60% negative; superseded by FTO Sec |
| Pronoun Rate | pronoun_rate.py | volume | per-token variant; using Pronouns per Second instead |
| Repetition Rate | repetition_rate.py | volume | per-token variant; using Repetitions per Second instead |
| Filler Word Rate | filler_word_rate.py | volume | per-token variant; using Filler Words per Second instead |
