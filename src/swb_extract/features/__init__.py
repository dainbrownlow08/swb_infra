"""Feature extractors, in two sub-packages (split 2026-08-20):

- ``inhouse/``    — the project's own extractors (every column in docs/FEATURES.md),
                    moved here verbatim from ``features/``; the shared helpers
                    (_text, _turn_index, _duration_lookup, word_align, backchannels)
                    live with them.
- ``thomas2018/`` — a replication of the eleven stylistic variables of Thomas,
                    Czerwinski, McDuff, Craswell & Mark (CHIIR 2018), built on the
                    in-house scaffolding and helpers.

Each sub-package lists its modules in ``EXTRACTORS``; ``swb-extract features <name>``
dispatches on a module's ``FEATURE_NAME``.
"""
