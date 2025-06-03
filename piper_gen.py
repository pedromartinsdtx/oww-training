import argparse
import logging
import random
import time
import wave
from pathlib import Path
from typing import List

import torch
from piper.voice import PiperVoice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePiperGenerator:
    def __init__(self, models: List[str], output_dir: str):
        self.models: List[str] = models
        self.voices: List[PiperVoice] = []

        use_cuda = torch.cuda.is_available()

        for model_path in self.models:
            # Generate audio using Piper ONNX model
            voice = PiperVoice.load(
                model_path=model_path,
                config_path=model_path + ".json",
                use_cuda=use_cuda,
            )  # If possible always use cuda. Dont know if it slows down things if a GPU isn't available
            self.voices.append(voice)

        self.output_dir: str = output_dir

    def generate_samples(
        self,
        texts: List[str],
        num_samples: int,
    ):
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            voice = random.choice(self.voices)
            # speaker_id = voice.speaker_id  # Get speaker ID if needed (for multi-speaker models)
            text = random.choice(texts)
            length_scale = round(
                random.triangular(0.5, 2, 1.0), 3
            )  # Controls the duration of the generated speech (larger = slower/longer)
            noise_scale = round(
                random.triangular(0.3, 1.2, 0.667), 3
            )  # Controls the amount of randomness/noise in generation (affects prosody)
            noise_w = round(
                random.triangular(0.3, 1.5, 0.8), 3
            )  # Controls pitch/energy variation (often for expressive TTS)

            logger.info(f"Generating sample {i + 1}/{num_samples} for text: {text}")
            synthesize_args = {
                # "speaker_id": speaker, # Specifies which speaker's voice to synthesize (multi-speaker models)
                "length_scale": length_scale,
                "noise_scale": noise_scale,
                "noise_w": noise_w,
            }

            # Save audio to file
            wav_path = (
                Path(self.output_dir)
                / f"{str(voice.config.espeak_voice)}_{text}_{time.monotonic_ns()}.wav"
            )
            with wave.open(str(wav_path), "wb") as wav_file:
                voice.synthesize(text, wav_file, **synthesize_args)


def main():
    parser = argparse.ArgumentParser(description="Simple Piper ONNX Sample Generator")

    parser.add_argument(
        "--models", nargs="+", required=True, help="Paths to Piper ONNX model files"
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        help="Text strings to convert to speech (alternative to --text-file)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        default="./samples_generated",
        type=str,
        help="Output directory for generated samples",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.texts:
        parser.error("--texts must be provided")

    # Load texts
    else:
        texts = args.texts
        logger.info(f"Using {len(texts)} provided texts")

    # Create generator and generate samples
    generator = SimplePiperGenerator(args.models, args.output_dir)
    generator.generate_samples(
        texts,
        args.num_samples,
    )


if __name__ == "__main__":
    main()
