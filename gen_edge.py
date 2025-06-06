import argparse
import asyncio
import random
import time
import warnings

import edge_tts

from utils.audio_files import mp3_to_wav, resample_audio_file_to_16k
from utils.play_audio import play_audio_file
from utils.remove_silence import remove_silence_from_file

warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")

BASE_OUTPUT_DIR = "samples/edge/clrs-edge"


async def text_to_speech(
    text, voice, rate: int = 0, pitch: int = 0, filename: str = None
):
    """Convert text to speech and save as an MP3 file."""
    if not text.strip():
        raise ValueError("Please enter text to convert.")
    if not voice:
        raise ValueError("Please select a voice.")
    if not filename:
        filename = "output.mp3"

    voice_short_name = voice.split(" - ")[0]
    rate_str = f"{rate:+d}%"
    pitch_str = f"{pitch:+d}Hz"

    communicate = edge_tts.Communicate(
        text, voice_short_name, rate=rate_str, pitch=pitch_str
    )

    await communicate.save(filename)

    # Convert MP3 to WAV and ensure 16kHz sample rate
    wav_filename = mp3_to_wav(filename, delete_mp3=True)
    resample_audio_file_to_16k(wav_filename, wav_filename, target_samplerate=16000)

    return wav_filename


async def generate_edge_tts_voice() -> str:
    """Main function to get user input and generate speech.

    Returns:
        str: The name of the generated audio file.
    """

    available_voices = [
        # "pt-BR-ThalitaMultilingualNeural",
        # "pt-BR-AntonioNeural",
        # "pt-BR-FranciscaNeural",
        "pt-PT-DuarteNeural",
        "pt-PT-RaquelNeural",
    ]

    print("Available Voices:")
    for voice in available_voices:
        print(f"{voice}")

    selected_voice = random.choice(available_voices)
    print(f"\nSelected Voice: {selected_voice}")

    text = "Clarisse."

    rate = int(
        random.triangular(-50, 0, 50)
    )  # More likely near 0, but possible extremes
    pitch = int(
        random.triangular(-20, 0, 20)
    )  # More likely near 0, but possible extremes
    print(f"Rate: {rate}%, Pitch: {pitch}Hz")

    time_id = time.strftime("%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"

    filename = f"{BASE_OUTPUT_DIR}-{selected_voice}-{time_id}.mp3"

    await text_to_speech(text, selected_voice, rate, pitch, filename=filename)

    print(f"Audio saved to: {filename.replace('.mp3', '.wav')}")

    return filename


def generate_edge_tts_voice_loop(
    iterations: int = 20,
    play_audio: bool = False,
    no_sleep: bool = False,
    remove_silence: bool = False,
):
    """Run the TTS generation in a loop."""
    generated_files = []

    for i in range(iterations):
        print(f"Voice Gen Edge TTS: (Attempt {i + 1})")
        try:
            filename = asyncio.run(generate_edge_tts_voice())
            print(f"Voice Gen Edge TTS: {i + 1} successful.")
            generated_files.append(filename)

            if play_audio and filename:
                print(f"Playing audio: {filename}")
                play_audio_file(filename)
            if not no_sleep:
                time.sleep(2)
        except Exception as e:
            print(f"Voice Gen Edge TTS: ERROR-(Attempt {i + 1}):\n{e}")
            if not no_sleep:
                time.sleep(5)

    # Remove silence from all generated files if requested
    if remove_silence and generated_files:
        print(f"\nRemoving silence from {len(generated_files)} generated files...")
        for filename in generated_files:
            try:
                print(f"Processing: {filename}")
                remove_silence_from_file(
                    filename, filename
                )  # Same input and output path to overwrite
                print(f"Silence removed from: {filename}")
            except Exception as e:
                print(f"Error removing silence from {filename}: {str(e)}")
        print("Silence removal completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate voice samples using edge TTS"
    )
    parser.add_argument(
        "--play-audio",
        action="store_true",
        help="Play audio files after generation (default: False)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations to run (default: 20)",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Disable sleep delays between iterations (default: False)",
    )
    parser.add_argument(
        "--remove-silence",
        action="store_true",
        help="Remove silence from all generated audio files after creation (default: False)",
    )

    args = parser.parse_args()
    generate_edge_tts_voice_loop(
        iterations=args.iterations,
        play_audio=args.play_audio,
        no_sleep=args.no_sleep,
        remove_silence=args.remove_silence,
    )
