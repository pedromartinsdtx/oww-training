# To run this code you need to install the following dependencies:
# pip install google-genai

import argparse
import json
import mimetypes
import os
import random
import struct
import time
from typing import Optional
import librosa

from dotenv import load_dotenv
from google import genai
import numpy as np
from google.genai import types

from utils.play_audio import play_audio_file

load_dotenv()


def _resample_audio(
    self, audio_data: np.ndarray, original_samplerate: int, target_samplerate: int
) -> np.ndarray:
    """
    Resamples audio data to the target sample rate.
    """
    if audio_data.ndim > 1 and audio_data.shape[1] == 1:
        audio_data = audio_data[:, 0]  # Convert to mono if needed

    resampled_audio = librosa.resample(
        y=audio_data, orig_sr=original_samplerate, target_sr=target_samplerate
    )
    return resampled_audio


def convert_bytes_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",  # ChunkID
        chunk_size,  # ChunkSize (total file size - 8 bytes)
        b"WAVE",  # Format
        b"fmt ",  # Subchunk1ID
        16,  # Subchunk1Size (16 for PCM)
        1,  # AudioFormat (1 for PCM)
        num_channels,  # NumChannels
        sample_rate,  # SampleRate
        byte_rate,  # ByteRate
        block_align,  # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",  # Subchunk2ID
        data_size,  # Subchunk2Size (size of audio data)
    )

    return header + audio_data


def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts:  # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass  # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass  # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


PROMPT = "Diz apenas a palavra 'Clarisse', em português de Portugal, como se estivesses a chamar um assistente virtual — no estilo de dizer 'Alexa' ou 'Siri'."

GEMINI_AVAILABLE_VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]


GEMINI_AVAILABLE_MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]

BASE_OUTPUT_DIR = "samples/gemini/clrs-gem"


def generate_gemini_voice(
    prompt: Optional[str] = PROMPT,
    model: Optional[str] = "gemini-2.5-flash-preview-tts",
    voice: Optional[str] = None,
    temperature: Optional[float] = 1.0,
    resample_to: Optional[int] = 16000,  # New argument for target sample rate
) -> str:
    # Arguments handling
    if voice is None:
        voice = random.choice(GEMINI_AVAILABLE_VOICES)

    # Connection to the TTS service
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        response_modalities=[
            "audio",
        ],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )

    file_name = None
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if (
            chunk.candidates[0].content.parts[0].inline_data
            and chunk.candidates[0].content.parts[0].inline_data.data
        ):
            time_id = (
                time.strftime("%m%d-%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"
            )
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                file_extension = ".wav"
                data_buffer = convert_bytes_to_wav(
                    inline_data.data, inline_data.mime_type
                )

            # --- Resample audio here ---
            # Try to load audio data as numpy array
            import io
            import soundfile as sf

            audio_np, orig_sr = sf.read(io.BytesIO(data_buffer), dtype="float32")
            # If mono, ensure shape is (N,)
            if audio_np.ndim > 1 and audio_np.shape[1] == 1:
                audio_np = audio_np[:, 0]

            # Resample if needed
            if resample_to and orig_sr != resample_to:
                audio_np = _resample_audio(None, audio_np, orig_sr, resample_to)
                # Save as WAV with new sample rate
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
                    sf.write(tmpf.name, audio_np, resample_to)
                    tmpf.seek(0)
                    data_buffer = tmpf.read()
                file_extension = ".wav"

            file_name = f"{BASE_OUTPUT_DIR}-{voice}-{time_id}{file_extension}"
            save_binary_file(file_name, data_buffer)
        else:
            print(
                f"!!!GEMINI_{voice}: No inline data found in chunk with prompt '{prompt}'."
            )
            print(chunk.text)

    return file_name


MAX_REQUESTS_PER_MINUTE = 3
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE  # = 20 seconds


def gemini_loop(iterations: int = 20, play_audio: bool = False):
    for i in range(iterations):
        print(f"Voice Gen Gemini: (Attempt {i + 1})")
        try:
            filename = generate_gemini_voice()
            print(f"Voice Gen Gemini: {i + 1} successful.")
            if play_audio and filename:
                print(f"Playing audio: {filename}")
                play_audio_file(filename)
            print(f"Waiting {REQUEST_INTERVAL} seconds for the next request...")
            time.sleep(REQUEST_INTERVAL)
        except Exception as e:
            if hasattr(e, "details"):
                print(json.dumps(e.details, indent=2))
            else:
                print(f"Error details not available: {e}")

            error_str = str(e)
            if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in error_str:
                print("Daily quota limit reached. Halting further requests.")
                break
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(error_str)
                sleep: int = 40
                print(
                    f"Rate limit hit. Waiting {sleep} seconds for the next request..."
                )
                time.sleep(sleep)
            else:
                print(f"Voice Gen Gemini: ERROR-(Attempt {i + 1}):\n{e}")
                time.sleep(5)


if __name__ == "__main__":
    """
    python3 -m services.gemini
    python3 gen_gemini.py --play-audio
    """

    parser = argparse.ArgumentParser(
        description="Generate voice samples using Gemini TTS"
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

    args = parser.parse_args()

    gemini_loop(iterations=args.iterations, play_audio=args.play_audio)
