#!/usr/bin/env python3
"""
Simplified Piper ONNX Sample Generator

A streamlined script to generate synthetic speech samples using Piper ONNX models.
Removes complexity around silence removal and augmentation to focus on core TTS generation.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from piper_phonemize import phonemize_espeak

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePiperGenerator:
    """Simple Piper ONNX model generator"""

    def __init__(
        self, model_paths: List[Union[str, Path]], output_dir: Union[str, Path]
    ):
        self.model_paths = [Path(p) for p in model_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load models and configs
        self.models = []
        self.configs = []
        self._load_models()

        # Setup resampler (22050 -> 16000 Hz)
        self.resampler = torchaudio.transforms.Resample(
            22050,
            16000,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_hann",
            beta=14.769656459379492,
        )

    def _load_models(self):
        """Load all ONNX models and their configurations"""
        print("Get available providers: ", ort.get_available_providers())
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )

        for model_path in self.model_paths:
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                continue

            if model_path.suffix.lower() != ".onnx":
                logger.error(f"Only ONNX models supported: {model_path}")
                continue

            # Load ONNX model
            try:
                model = ort.InferenceSession(str(model_path), providers=providers)
                self.models.append(model)
                logger.info(f"Loaded model: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
                continue

            # Load config
            config_path = f"{model_path}.json"
            if not Path(config_path).exists():
                logger.error(f"Config not found: {config_path}")
                continue

            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.configs.append(config)
                logger.info(f"Loaded config: {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config {config_path}: {e}")
                continue

        if not self.models:
            raise ValueError("No valid models loaded!")

        logger.info(f"Successfully loaded {len(self.models)} models")

    def _get_phonemes(self, text: str, voice: str, phoneme_id_map: dict) -> List[int]:
        """Convert text to phoneme IDs"""
        # Get phonemes from espeak
        phonemes = [
            p
            for sentence_phonemes in phonemize_espeak(text, voice)
            for p in sentence_phonemes
        ]

        # Build phoneme ID sequence
        phoneme_ids = list(phoneme_id_map["^"])  # Beginning of utterance
        phoneme_ids.extend(phoneme_id_map["_"])  # Separator

        # Add phonemes
        for phoneme in phonemes:
            if phoneme in phoneme_id_map:
                phoneme_ids.extend(phoneme_id_map[phoneme])
            else:
                logger.warning(f"Unknown phoneme: {phoneme}")

        phoneme_ids.extend(phoneme_id_map["$"])  # End of utterance

        return phoneme_ids

    def _generate_with_onnx(
        self,
        model,
        phoneme_ids: List[int],
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
    ) -> np.ndarray:
        """Generate audio using ONNX model"""
        # Prepare inputs
        x = np.array([phoneme_ids], dtype=np.int64)
        x_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_scale_w], dtype=np.float32)

        # Run inference
        inputs = {
            "input": x,
            "input_lengths": x_lengths,
            "scales": scales,
        }

        outputs = model.run(None, inputs)
        # Flatten the audio output to 1D array
        audio = outputs[0]
        while len(audio.shape) > 1:
            audio = audio.squeeze(0)
        return audio

    def generate_samples(
        self,
        texts: List[str],
        num_samples: int,
        noise_scales: List[float] = None,
        length_scales: List[float] = None,
        noise_scale_ws: List[float] = None,
    ) -> None:
        """Generate speech samples"""

        # Default parameter ranges
        if noise_scales is None:
            noise_scales = [0.6, 0.667, 0.7, 0.8]
        if length_scales is None:
            length_scales = [0.8, 1.0, 1.2]
        if noise_scale_ws is None:
            noise_scale_ws = [0.7, 0.8, 0.9]

        logger.info(f"Generating {num_samples} samples using {len(self.models)} models")

        sample_count = 0

        while sample_count < num_samples:
            # Select random text and model
            text = random.choice(texts)
            model_idx = random.randint(0, len(self.models) - 1)
            model = self.models[model_idx]
            config = self.configs[model_idx]

            # Get voice and phoneme mapping
            voice = config["espeak"]["voice"]
            phoneme_id_map = config["phoneme_id_map"]

            # Random parameters
            noise_scale = random.choice(noise_scales)
            length_scale = random.choice(length_scales)
            noise_scale_w = random.choice(noise_scale_ws)

            try:
                # Convert text to phonemes
                phoneme_ids = self._get_phonemes(text, voice, phoneme_id_map)

                # Generate audio
                audio = self._generate_with_onnx(
                    model, phoneme_ids, noise_scale, length_scale, noise_scale_w
                )

                # Ensure audio is 1D
                audio = np.squeeze(audio)
                if len(audio.shape) != 1:
                    raise ValueError(f"Expected 1D audio, got shape: {audio.shape}")

                # Get sample rate from model config
                sample_rate = config["audio"]["sample_rate"]

                # Normalize to int16 range
                audio_norm = self._normalize_audio(audio)

                # Save audio file
                model_name = self.model_paths[model_idx].stem
                filename = f"sample_{sample_count:06d}_{model_name}.wav"
                output_path = self.output_dir / filename

                sf.write(output_path, audio_norm, sample_rate)

                logger.info(
                    f"Generated sample {sample_count + 1}/{num_samples}: {filename}"
                )
                sample_count += 1

            except Exception as e:
                logger.error(f"Failed to generate sample: {e}")
                continue

        logger.info(
            f"Successfully generated {sample_count} samples in {self.output_dir}"
        )

    def _normalize_audio(
        self, audio: np.ndarray, max_val: float = 32767.0
    ) -> np.ndarray:
        """Normalize audio and convert to int16"""
        if len(audio) == 0:
            return np.array([], dtype=np.int16)

        # Piper models typically output float32 in range [-1, 1]
        # Ensure audio is in the expected range
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            # Clip to [-1, 1] range if needed
            audio = np.clip(audio, -1.0, 1.0)
            # Scale to int16 range
            audio_norm = audio * max_val
        else:
            # If it's already integer, normalize differently
            current_max = np.max(np.abs(audio))
            if current_max > 0:
                audio_norm = audio * (max_val / current_max)
            else:
                audio_norm = audio

        # Ensure we don't exceed int16 range
        audio_norm = np.clip(audio_norm, -max_val, max_val)
        return audio_norm.astype(np.int16)


def load_texts_from_file(file_path: str) -> List[str]:
    """Load texts from a file (one per line)"""
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


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
        "--text-file",
        help="Path to file containing texts (one per line, alternative to --texts)",
    )
    parser.add_argument(
        "--num-samples", type=int, required=True, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        default="./generated_samples",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--noise-scales",
        nargs="+",
        type=float,
        default=[0.6, 0.667, 0.7, 0.8],
        help="Noise scale values to randomly choose from",
    )
    parser.add_argument(
        "--length-scales",
        nargs="+",
        type=float,
        default=[0.8, 1.0, 1.2],
        help="Length scale values to randomly choose from",
    )
    parser.add_argument(
        "--noise-scale-ws",
        nargs="+",
        type=float,
        default=[0.7, 0.8, 0.9],
        help="Noise scale W values to randomly choose from",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.texts and not args.text_file:
        parser.error("Either --texts or --text-file must be provided")

    if args.texts and args.text_file:
        parser.error("Cannot use both --texts and --text-file")

    # Load texts
    if args.text_file:
        texts = load_texts_from_file(args.text_file)
        logger.info(f"Loaded {len(texts)} texts from {args.text_file}")
    else:
        texts = args.texts
        logger.info(f"Using {len(texts)} provided texts")

    if not texts:
        logger.error("No texts provided!")
        return

    # Create generator and generate samples
    generator = SimplePiperGenerator(args.models, args.output_dir)
    generator.generate_samples(
        texts,
        args.num_samples,
        args.noise_scales,
        args.length_scales,
        args.noise_scale_ws,
    )


if __name__ == "__main__":
    main()
