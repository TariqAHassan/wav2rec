import librosa
import soundfile

from tests import (
    GENERATED_WAVEFORMS_EXT,
    GENERATED_WAVEFORMS_LENGTH,
    GENERATED_WAVEFORMS_PATH,
    GENERATED_WAVEFORMS_SR,
)

if __name__ == "__main__":
    GENERATED_WAVEFORMS_PATH.mkdir(parents=True, exist_ok=True)

    for tone in (440, 480, 520):
        y = librosa.tone(
            tone,
            sr=GENERATED_WAVEFORMS_SR,
            length=GENERATED_WAVEFORMS_LENGTH,
        )

        soundfile.write(
            GENERATED_WAVEFORMS_PATH.joinpath(f"tone_{tone}.{GENERATED_WAVEFORMS_EXT}"),
            data=y,
            samplerate=GENERATED_WAVEFORMS_SR,
            subtype="PCM_24",
        )
