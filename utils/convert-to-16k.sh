#!/bin/bash

# Script to convert .wav files in a folder to 16kHz sampling rate
# and then remove the original files.

# Check if an input directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_directory>"
  exit 1
fi

INPUT_DIR="$1"
# Output directory will be a sibling to the input, not inside it,
# to avoid issues if the script is run multiple times or if input is "."
PARENT_DIR=$(dirname "$INPUT_DIR")
BASE_NAME=$(basename "$INPUT_DIR")
OUTPUT_DIR_NAME="${BASE_NAME}_16k"
OUTPUT_DIR="${PARENT_DIR}/${OUTPUT_DIR_NAME}"


# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory '$INPUT_DIR' not found."
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Input directory: $INPUT_DIR"
echo "Output directory for converted files: $OUTPUT_DIR"
echo "WARNING: Original .wav files will be DELETED after successful conversion."
read -p "Do you want to proceed? (yes/no): " confirmation
if [[ "$confirmation" != "yes" ]]; then
  echo "Operation cancelled by user."
  exit 0
fi

echo "Converting .wav files to 16kHz and removing originals..."

# Find all .wav files in the input directory and its subdirectories
find "$INPUT_DIR" -type f -name "*.wav" | while read -r wav_file; do
  # Determine the relative path of the file with respect to INPUT_DIR
  relative_path="${wav_file#$INPUT_DIR/}"
  # Define the output file path, creating subdirectories in OUTPUT_DIR as needed
  output_file_path="${OUTPUT_DIR}/${relative_path}"
  output_file_dir=$(dirname "$output_file_path")

  # Create the subdirectory structure in the output directory
  mkdir -p "$output_file_dir"

  echo "Processing: $wav_file -> $output_file_path"
  # Use ffmpeg to convert the .wav file to 16kHz
  ffmpeg -i "$wav_file" -ar 16000 -y "$output_file_path" > /dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "Successfully converted: $output_file_path"
    # Remove the original file
    rm "$wav_file"
    if [ $? -eq 0 ]; then
      echo "Successfully removed original: $wav_file"
    else
      echo "Error removing original: $wav_file"
    fi
  else
    echo "Error converting: $wav_file. Original file will not be removed."
  fi
done

echo "Conversion and removal process complete."
echo "Converted files are in: $OUTPUT_DIR"
echo "If all files were processed, the original directory '$INPUT_DIR' might now only contain non-.wav files or be empty if it only contained .wav files."