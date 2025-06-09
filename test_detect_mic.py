import argparse
import os
import threading
import time
import wave
from collections import deque

import numpy as np
import pyaudio
import pygame
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
    "--detection-sound",
    help="The sound to play when wakeword is detected",
    type=str,
    default="ww_check.mp3",
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

# Load openwakeword model
if args.model_path != "":
    owwModel = Model(
        wakeword_models=[args.model_path],
        inference_framework=args.inference_framework,
        # enable_speex_noise_suppression=not args.no_noise_suppression,
    )
else:
    wakeword_models = [
        os.path.join("models", f) for f in os.listdir("models") if f.endswith(".tflite")
    ]
    owwModel = Model(
        # wakeword_models=wakeword_models,
        wakeword_models=[
            # "alexa_v0.1.tflite",
            # "models-ww/Clarisse_v-piper.onnx",
            # "models-ww/Clarisse_v1.2-piper.onnx",
            # "models-ww/Clarisse_v2_piper.onnx",
            # "models-ww/Clarisse_v2.5_piper.onnx",
            #
            # "models-ww/Hey_Clariss_v1_piper.onnx",
            # "models-ww/Hey_Clariss_v1.2_piper.onnx",
            # "models-ww/Hey_Clariss_v2_piper.onnx",
            #
            "models-ww/Olá_Clãriss-v1-piper.onnx",
            "models-ww/olá_cleddeess-v2.onnx",
            "models-ww/olá_cledeess-v3.onnx",
            "models-ww/holá_cleddeess.onnx",
            "models-ww/olá_cleddeess.onnx",
        ],
        inference_framework="onnx",
        # inference_framework="tflite",
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

    os.makedirs("audios/logs", exist_ok=True)
    filepath = os.path.join("audios/logs", filename)

    audio_data = np.array(list(audio_buffer), dtype=np.int16)

    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(audio.get_sample_size(FORMAT))
        wav_file.setframerate(RATE)
        wav_file.writeframes(audio_data.tobytes())

    return filepath


def play_sound():
    def play_sound_thread():
        try:
            pygame.mixer.music.load(f"audios/{args.detection_sound}")
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error playing sound: {e}")

    sound_thread = threading.Thread(target=play_sound_thread)
    sound_thread.daemon = True
    sound_thread.start()


def handle_wakeword_detection(score):
    global last_detection_time
    current_time = time.time()

    if current_time - last_detection_time < 2:
        return

    last_detection_time = current_time

    play_sound()
    save_audio_buffer(score)


def suppress_alsa_errors():
    # Suppress ALSA errors by redirecting stderr temporarily
    import ctypes

    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
    )

    def py_error_handler(filename, line, function, err, fmt):
        pass

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so")
        asound.snd_lib_error_set_handler(c_error_handler)
    except Exception:
        pass


if __name__ == "__main__":
    # suppress_alsa_errors()
    pygame.mixer.init()  # Initialize mixer once at the start

    print("\n\n")
    print("#" * 100)
    print("Listening for wakewords...")
    print("#" * 100)
    print("\n" * (n_models * 3))

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
