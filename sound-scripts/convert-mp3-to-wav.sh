#!/bin/bash

# Usage: ./convert-mp3-to-wav.sh [directory]
# Convert all .mp3 files in the specified directory (or current directory if not specified) to .wav using ffmpeg

DIR="${1:-.}"
DIR="${DIR%/}" # Remove trailing slash if any

for file in "$DIR"/*.mp3; do
    [ -e "$file" ] || continue
    base="${file%.mp3}"
    if ffmpeg -y -i "$file" "${base}.wav"; then
        echo "Converted $file to ${base}.wav"
        rm "$file"
        echo "Deleted $file"
    else
        echo "Failed to convert $file"
    fi
done