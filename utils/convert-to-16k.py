import argparse
import os

import librosa
import soundfile as sf

# Define supported audio formats
SUPPORTED_FORMATS = ".wav"


def resample_audio_to_16khz_librosa(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                file_path = os.path.join(root, file)
                try:
                    # Load audio with original sample rate
                    audio, sr = librosa.load(file_path, sr=None)

                    if sr != 16000:
                        print(f"Resampling: {file_path} ({sr} Hz → 16000 Hz)")

                        # Resample audio
                        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

                        # Save resampled audio
                        sf.write(file_path, audio_16k, 16000)
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
        resample_audio_to_16khz_librosa(args.folder)
    else:
        print("Invalid folder path.")
