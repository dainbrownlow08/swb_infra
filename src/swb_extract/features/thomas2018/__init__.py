"""The eleven stylistic variables of Thomas, Czerwinski, McDuff, Craswell & Mark (2018),
"Style and Alignment in Information-Seeking Conversation", CHIIR '18, §3.2
(https://doi.org/10.1145/3176349.3176388) — "the MSFT paper" of AUDIT.md §4E-h —
rebuilt on Switchboard with the in-house scaffolding.

The paper derives, "for each participant in each task, eleven variables in six
categories" (names are the paper's, verbatim):

  People               ppron   rate of 1st/2nd-person pronouns (not 3rd)
  Rate of speech       wps     words / total duration of utterances (a micro-average)
  Pauses, turn-taking  wpu     mean words per utterance
                       wpp     words / number of between-own pauses (overall average)
                       boplen  mean length of between-own pauses
                       poplen  mean length of post-other pauses
  Expressive phonology pv      variance of F0 over frames with a speech signal
                       lv      loudness variance, "measured the same way"
  Overlap              olap    proportion of utterances that begin while the partner talks
  Re-statement         rept    mean # terms repeated from the same person's previous utt
                       repu    fraction of utterances with >= 1 repeated term

"A between-own pause is a period during a single utterance where there is no speech
signal (periods where OpenSMILE reports no F0), and a post-other pause is the gap
between the end of one participant's utterance and the start of their partner's next."
"An 'utterance' here is simply a line of transcript."

Unit of analysis. The paper's unit is participant x task; on Switchboard that is one
conversation side (call, A|B). The pipeline's unit is the utterance, so each module
writes the per-utterance INGREDIENTS of its variable (manifest order, blank = not
measurable) and exposes ``aggregate(rows)`` — the paper's statistic over any group of
utterance rows (a side, a caller, a minute): ratio of sums / pooled variance / mean,
never a mean of per-utterance ratios. ``side_table.py`` applies the eleven aggregates
per conversation side (``swb-extract thomas2018-side``). Ingredient columns:

  ppron    ppron                   1st/2nd-person pronoun count      -> sum / sum(wpu)
  wps      wps sec                 utterance duration, word-tight     -> sum(wpu) / sum
  wpu      wpu                     word count                         -> mean
  wpp      wpp pauses              between-own pause count            -> sum(wpu) / sum
  boplen   boplen n, boplen sec    pause count, total pause seconds   -> sum(sec) / sum(n)
  poplen   poplen                  post-other pause (s); blank if none -> mean
  pv       pv n, pv mean, pv var   voiced frames, F0 mean/var (Hz, Hz^2) -> pooled variance
  lv       lv n, lv mean, lv var   same for frame RMS                  -> pooled variance
  olap     olap                    0/1                                 -> mean
  rept     rept                    repeated-term count; blank if no previous own utt -> mean
  repu     repu                    0/1; blank likewise                 -> mean

Corpus-mapping decisions, each documented where it bites: utterance start/end are
word-tight (``_bounds``); "speech signal" = librosa.pyin voicing on the in-house pitch
settings, loudness = frame RMS on the same grid (``_prosody``; OpenSMILE is not a
dependency here); stopwords + stemming for rept/repu = a frozen NLTK list + Porter
(``_terms``). Nothing is z-scored here — the paper rescales the eleven to zero mean /
unit SD before summing them with the Table 2 loadings (``PAPER_LOADINGS``) into its
"involvement" score.
"""
from . import boplen, lv, olap, poplen, ppron, pv, rept, repu, wpp, wps, wpu

# Paper order (§3.2 / Table 2). side_table writes the variables in this order.
EXTRACTORS = (ppron, wps, wpu, wpp, boplen, poplen, pv, lv, olap, rept, repu)
VARIABLES = tuple(m.FEATURE_NAME for m in EXTRACTORS)

# Table 2: loading of each (z-scored) variable on the first principal component of
# the MISC data — the paper's "involvement" — i.e. the signs it found. "The sign of
# each is consistent with predictions, except overlap ratio", whose −0.01 the paper
# reads as carrying no signal.
PAPER_LOADINGS = {
    "ppron": 0.13,
    "wps": 0.39,
    "wpu": 0.45,
    "wpp": 0.44,
    "boplen": -0.10,
    "poplen": -0.27,
    "pv": 0.09,
    "lv": 0.21,
    "olap": -0.01,
    "rept": 0.39,
    "repu": 0.39,
}
