from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO / "swb_ms98_transcriptions_cleaned"
AUDIO = REPO / "audio"

GOLDEN_TRANSCRIPT = TRANSCRIPTS / "20" / "2001" / "sw2001A-ms98-a-trans.text"
GOLDEN_AUDIO = AUDIO / "disc1" / "data" / "sw02001.A.wav"


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def golden_transcript() -> Path:
    if not GOLDEN_TRANSCRIPT.is_file():
        pytest.skip(f"missing golden transcript: {GOLDEN_TRANSCRIPT}")
    return GOLDEN_TRANSCRIPT


@pytest.fixture(scope="session")
def golden_audio() -> Path:
    if not GOLDEN_AUDIO.is_file():
        pytest.skip(f"missing golden audio: {GOLDEN_AUDIO}")
    return GOLDEN_AUDIO


@pytest.fixture(scope="session")
def audio_root() -> Path:
    if not AUDIO.is_dir():
        pytest.skip(f"missing audio root: {AUDIO}")
    return AUDIO


@pytest.fixture(scope="session")
def transcript_root() -> Path:
    if not TRANSCRIPTS.is_dir():
        pytest.skip(f"missing transcript root: {TRANSCRIPTS}")
    return TRANSCRIPTS
