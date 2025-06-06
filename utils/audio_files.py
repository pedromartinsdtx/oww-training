# To run this code you need to install the following dependencies:
# pip install pydub
# pip install audioop-lts


from pathlib import Path
from typing import Optional

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




if __name__ == "__main__":
    for audio_file in find_audio_files_with_min_duration(".", 2):
        print(f"Found audio file: {audio_file} with duration > 2 seconds")
