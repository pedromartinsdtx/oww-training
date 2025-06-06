# To run this code you need to install the following dependencies:
# pip install pydub
# pip install audioop-lts


import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import mediainfo

audio_extensions = [".wav"]


def find_audio_files_with_min_duration(directory=".", min_duration=2) -> list[str]:
    """
    Yields audio files in the given directory (recursively) with duration greater than min_duration seconds.
    """
    for file in Path(directory).rglob("*"):
        if file.suffix.lower() in audio_extensions:
            info = mediainfo(str(file))
            duration = round(float(info["duration"]))
            if duration > min_duration:
                yield file


def mp3_to_wav(
    mp3_path: str, wav_path: Optional[str] = None, delete_mp3: Optional[bool] = True
) -> str:
    """
    Converts an MP3 file to WAV format.

    Parameters:
    - mp3_path: Path to the input MP3 file.
    - wav_path: Path to save the output WAV file. If None, saves with the same name as mp3_path but with .wav extension.
    - delete_mp3: If True, deletes the original MP3 file after successful conversion.
    """
    mp3_path = Path(mp3_path)
    if wav_path is None:
        wav_path = mp3_path.with_suffix(".wav")
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    if delete_mp3 and wav_path.exists():
        mp3_path.unlink()

    return str(wav_path)


def resample_audio_to_16k(
    audio_data: np.ndarray, original_samplerate: int, target_samplerate: int = 16000
) -> np.ndarray:
    """
    Resamples audio data to 16 kHz.
    """
    if audio_data.ndim > 1 and audio_data.shape[1] == 1:
        audio_data = audio_data[:, 0]  # Convert to mono if needed

    resampled_audio = librosa.resample(
        y=audio_data, orig_sr=original_samplerate, target_sr=target_samplerate
    )
    return resampled_audio


def resample_audio_file_to_16k(
    input_filepath: str, output_filepath: str = None, target_samplerate: int = 16000
) -> str:
    """
    Resamples an audio file to 16 kHz and saves the output.

    Parameters:
        input_filepath (str): Path to the input audio file.
        output_filepath (str, optional): Path to save the resampled file. If None, appends '_16k.wav' to the original filename.

    Returns:
        str: Path to the resampled audio file.
    """
    # Load audio
    audio_data, orig_sr = librosa.load(input_filepath, sr=None, mono=True)

    # Resample to 16 kHz
    resampled_audio = librosa.resample(
        audio_data, orig_sr=orig_sr, target_sr=target_samplerate
    )

    # Define output file path
    if output_filepath is None:
        base, _ = os.path.splitext(input_filepath)
        output_filepath = f"{base}_16k.wav"

    # Save resampled audio
    sf.write(
        output_filepath, resampled_audio, samplerate=target_samplerate, format="WAV"
    )

    return output_filepath


if __name__ == "__main__":
    for audio_file in find_audio_files_with_min_duration(".", 2):
        print(f"Found audio file: {audio_file} with duration > 2 seconds")
