"""

    Tests

"""
from pathlib import Path

GENERATED_WAVEFORMS_SR: int = 22_050
GENERATED_WAVEFORMS_DURATION: int = 4
GENERATED_WAVEFORMS_LENGTH: int = GENERATED_WAVEFORMS_SR * GENERATED_WAVEFORMS_DURATION
GENERATED_WAVEFORMS_EXT: str = "wav"
GENERATED_WAVEFORMS_PATH = Path("tests/_fixtures/waveforms").absolute()