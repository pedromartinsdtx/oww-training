#!/bin/bash
# reqs:
# sudo apt install -y portaudio19-dev
# sudo apt install -y pulseaudio pulseaudio-utils

#* This script finds and plays all audio files in the current directory (or specified path) that match a given prefix or pattern.

# Signal handler to exit gracefully
cleanup() {
    echo ""
    echo "Interrupted! Stopping playback..."
    # Kill any running paplay processes
    pkill -f paplay
    exit 130
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

if [ $# -eq 0 ]; then
    echo "Usage: $0 [prefix|directory|path_prefix]"
    echo "  prefix: search for files in current directory matching the prefix"
    echo "  directory: absolute path to a directory to search in"
    echo "  path_prefix: absolute path prefix to match files"
    echo
fi

PREFIX="${1:-}"

echo "Finding audio files..."

# Determine search directory and filename pattern
if [ -z "$PREFIX" ]; then
    search_dir="./"
    filename_pattern="*"
elif [ -d "$PREFIX" ]; then
    search_dir="$PREFIX"
    filename_pattern="*"
elif [[ "$PREFIX" == */* ]]; then
    search_dir=$(dirname "$PREFIX")
    filename_pattern="$(basename "$PREFIX")*"
else
    search_dir="./"
    filename_pattern="${PREFIX}*"
fi

# Verify search directory exists
if [ ! -d "$search_dir" ]; then
    echo "Directory not found: $search_dir"
    exit 1
fi

# Find audio files
audio_files=($(find "$search_dir" -maxdepth 1 -type f -name "$filename_pattern" \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" \) | sort))

# Check if any audio files were found
if [ ${#audio_files[@]} -eq 0 ]; then
    echo "No audio files found in the search location."
    exit 1
fi

echo "Found ${#audio_files[@]} audio file(s):"
for file in "${audio_files[@]}"; do
    echo "  - $(basename "$file")"
done
echo ""

echo "Playing audio files..."

# Loop through each file and play it
for file in "${audio_files[@]}"; do
    if [ -f "$file" ]; then
        # echo "Playing: $file"
        aplay "$file"
        echo "---"
    else
        echo "File not found: $file"
    fi
done

echo "All audio files have been played."