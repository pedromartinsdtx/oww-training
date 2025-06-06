import os
from pydub import AudioSegment
import argparse

# Define supported audio formats
SUPPORTED_FORMATS = ".wav"


def resample_audio_to_16khz(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                file_path = os.path.join(root, file)
                try:
                    # Load audio
                    audio = AudioSegment.from_file(file_path)

                    if audio.frame_rate != 16000:
                        print(
                            f"Resampling: {file_path} ({audio.frame_rate} Hz → 16000 Hz)"
                        )

                        # Resample to 16kHz
                        audio_16k = audio.set_frame_rate(16000)

                        # Temporary file path with same extension
                        temp_path = file_path + ".temp"

                        # Export and replace original
                        audio_16k.export(
                            temp_path, format=os.path.splitext(file_path)[1][1:]
                        )
                        os.remove(file_path)
                        os.rename(temp_path, file_path)
                    else:
                        print(f"Already 16kHz: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample all .wav files in a folder to 16kHz."
    )
    parser.add_argument("folder", help="Path to the folder containing audio files")
    args = parser.parse_args()

    if os.path.isdir(args.folder):
        resample_audio_to_16khz(args.folder)
    else:
        print("Invalid folder path.")
