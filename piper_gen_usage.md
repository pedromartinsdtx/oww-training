22# Simple Piper ONNX Generator Usage

This script provides a simplified way to generate speech samples using Piper ONNX models.

## Basic Usage

```bash
# Generate 10 samples using a single model with text from command line
python piper_gen.py \
    --models /path/to/model.onnx \
    --texts "Hello world" "How are you today?" "This is a test" \
    --num-samples 10 \
    --output-dir ./samples

# Generate 50 samples using multiple models with text from a file
python piper_gen.py \
    --models /path/to/model1.onnx /path/to/model2.onnx \
    --text-file texts.txt \
    --num-samples 50 \
    --output-dir ./output

# Generate with custom parameters
python piper_gen.py \
    --models /path/to/model.onnx \
    --texts "Hello" "World" \
    --num-samples 20 \
    --noise-scales 0.6 0.7 0.8 \
    --length-scales 0.9 1.0 1.1 \
    --noise-scale-ws 0.7 0.8 \
    --output-dir ./custom_output
```

## Example with Available Models

Using models from the piper-sample-generator directory:

```bash
# Generate 20 samples with Portuguese models
python piper_gen.py \
    --models ../piper-sample-generator/models/pt_PT-tugao-medium.onnx \
             ../piper-sample-generator/models/pt_PT-rita.onnx \
    --texts "Olá mundo" "Como estás hoje?" "Este é um teste" "Boa tarde" \
    --num-samples 20 \
    --output-dir ./portuguese_samples
```

## Text File Format

Create a text file (e.g., `texts.txt`) with one sentence per line:

```
Hello, how are you today?
This is a sample sentence.
The weather is nice today.
I love using text-to-speech technology.
```

## Parameters

- `--models`: List of ONNX model paths (required)
- `--texts`: Text strings to convert (alternative to --text-file)
- `--text-file`: File with texts, one per line (alternative to --texts)
- `--num-samples`: Number of samples to generate (required)
- `--output-dir`: Output directory (default: ./generated_samples)
- `--noise-scales`: Noise scale values to randomly choose from
- `--length-scales`: Length scale values (affects speech speed)
- `--noise-scale-ws`: Noise scale W values (affects word timing variation)

## Output

The script generates WAV files at 16kHz with the naming pattern:
`sample_XXXXXX_modelname.wav`

Where:
- `XXXXXX` is a 6-digit sample number
- `modelname` is the name of the ONNX model used

## Requirements

Install the required packages:

```bash
pip install numpy onnxruntime soundfile torch torchaudio piper-phonemize
```

Note: The script automatically detects CUDA availability and uses GPU acceleration when possible.
