import argparse
import os
import time
import wave
from collections import deque

import numpy as np
import pyaudio
import simpleaudio as sa
from openwakeword.model import Model
from scipy import signal

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False,
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="",
    required=False,
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default="tflite",
    required=False,
)
parser.add_argument(
    "--detection_sound",
    help="The sound to play when wakeword is detected",
    type=str,
    default="ww_check.wav",
    required=False,
)
parser.add_argument(
    "--no-noise-suppression",
    help="Disable Speex noise suppression. Noise suppression is enabled by default.",
    action="store_true",  # Use action='store_true' for a boolean flag
    required=False,
)

args = parser.parse_args()

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
TARGET_RATE = 16000
CHUNK = args.chunk_size
downsample_ratio = RATE // TARGET_RATE

# Initialize PyAudio
audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    # input_device_index=2, #! É preciso esta linha no Raspberry Pi para funcionar
)

WW_MODELS_FOLDER = "models-ww"
CLARISSE_MODELS = f"{WW_MODELS_FOLDER}/clarisse"
HEY_CLARISSE_MODELS = f"{WW_MODELS_FOLDER}/hey-clarisse"
OLA_CLARISSE_MODELS = f"{WW_MODELS_FOLDER}/ola-clarisse"
PARA_MODELS = f"{WW_MODELS_FOLDER}/para"

# Load openwakeword model
if args.model_path != "":
    owwModel = Model(
        wakeword_models=[args.model_path],
        inference_framework=args.inference_framework,
        # enable_speex_noise_suppression=not args.no_noise_suppression,
    )
else:
    inference_framework = "tflite"
    models_path = PARA_MODELS
    wakeword_models = [
        os.path.join(models_path, f)
        for f in os.listdir(models_path)
        if f.endswith(f".{inference_framework}")
    ]

    owwModel = Model(
        # wakeword_models=[
        #     "alexa_v0.1.tflite",
        # ],
        wakeword_models=wakeword_models,
        inference_framework=inference_framework,
        # enable_speex_noise_suppression=not args.no_noise_suppression,
    )

n_models = len(owwModel.models.keys())

# Global variables
last_detection_time = 0
BUFFER_DURATION = 3
BUFFER_SIZE = RATE * BUFFER_DURATION
audio_buffer = deque(maxlen=BUFFER_SIZE)


def save_audio_buffer(score, filename=None):
    if filename is None:
        timestamp = time.strftime("%m%d_%H%M%S")
        filename = f"ww_{timestamp}_{score}.wav"

    os.makedirs("logs/test-detect-mic", exist_ok=True)
    filepath = os.path.join("logs/test-detect-mic", filename)

    audio_data = np.array(list(audio_buffer), dtype=np.int16)

    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(audio.get_sample_size(FORMAT))
        wav_file.setframerate(RATE)
        wav_file.writeframes(audio_data.tobytes())

    return filepath


def play_audio_file(audio_file: str, wait: bool = False):
    try:
        if not audio_file.lower().endswith(".wav"):
            print(
                "SimpleAudio doesn't natively support MP3. Converting or using a WAV file is recommended."
            )
            return

        wave_obj = sa.WaveObject.from_wave_file(audio_file)
        play_obj = wave_obj.play()

        if wait:
            play_obj.wait_done()
    except Exception as e:
        print(f"Error playing sound from file {audio_file}: {e}")


def handle_wakeword_detection(score):
    global last_detection_time
    current_time = time.time()

    if current_time - last_detection_time < 2:
        return

    last_detection_time = current_time

    play_audio_file(f"audio_samples/{args.detection_sound}")
    save_audio_buffer(score)


if __name__ == "__main__":
    print("\n\n")
    print("#" * 100)
    print("Listening for wakewords...")
    print("#" * 100)
    print("\n" * (n_models * 3))

    # Create a file to log wakeword triggers
    os.makedirs("logs", exist_ok=True)
    ww_trigger_log_path = os.path.join("logs", "ww_triggers.txt")

    with open(ww_trigger_log_path, "w") as log_file:
        log_file.write("Wakeword Trigger Log\n")
        log_file.write("=" * 50 + "\n")

    while True:
        audio_data_raw = mic_stream.read(CHUNK, exception_on_overflow=False)
        audio_data_np = np.frombuffer(audio_data_raw, dtype=np.int16)

        # Add raw audio to buffer for saving
        audio_buffer.extend(audio_data_np)

        audio_float = audio_data_np.astype(np.float32)
        downsampled = signal.decimate(audio_float, downsample_ratio, ftype="iir")
        audio_16k = downsampled.astype(np.int16)

        prediction = owwModel.predict(audio_16k)

        n_spaces = 16
        output_string_header = (
            "Model Name         | Score | Wakeword Status\n"
            "--------------------------------------\n"
        )

        for mdl in owwModel.prediction_buffer.keys():
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], ".3f").replace("-", "")

            if scores[-1] > 0.5:
                wakeword_status = "Wakeword Detected!"
                with open(ww_trigger_log_path, "a") as log_file:
                    log_entry = (
                        f"Model: {mdl}, Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    with open(ww_trigger_log_path, "r+") as log_file:
                        existing_entries = log_file.readlines()
                        if log_entry not in existing_entries:
                            log_file.write(log_entry)
                handle_wakeword_detection(curr_score)

            else:
                wakeword_status = "--"

            output_string_header += f"{mdl:<20}| {curr_score:<5}| {wakeword_status}\n"

        # Clear the terminal before printing
        print("\033c", end="")  # ANSI clear screen

        print("#" * 100)
        print("Listening for wakewords...")
        print("#" * 100)
        print(output_string_header)
