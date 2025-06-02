import argparse
import os
import tempfile
import shutil

from pydub import AudioSegment
from pydub.silence import split_on_silence


def remove_silence_from_file(input_path, output_path=None):
    """
    Remove silence from a single audio file.

    Args:
        input_path (str): Path to the input audio file
        output_path (str, optional): Path for the output file. If None, adds '_no_silence' suffix

    Returns:
        str: Path to the processed audio file
    """
    # Load the audio file
    sound = AudioSegment.from_file(input_path)

    # Analyze audio to determine appropriate silence threshold
    average_loudness = sound.dBFS

    # Use average loudness minus 14dB as a good starting point for silence threshold
    silence_thresh = average_loudness - 14

    # Split the audio on silence
    chunks = split_on_silence(
        sound,
        min_silence_len=400,  # slightly more sensitive to shorter silences
        silence_thresh=silence_thresh,
        keep_silence=150,  # keep a short gap to make speech natural
    )

    # Combine chunks back into one audio segment
    processed_sound = AudioSegment.empty()
    for chunk in chunks:
        processed_sound += chunk

    # Generate output path if not provided
    if output_path is None:
        base_name, ext = os.path.splitext(input_path)
        output_path = f"{base_name}_no_silence{ext}"

    # Export the processed audio
    processed_sound.export(output_path, format="wav")

    return output_path


def process_folder(folder_path, output_folder=None):
    """
    Process all audio files in a folder to remove silence.

    Args:
        folder_path (str): Path to the folder containing audio files
        output_folder (str, optional): Path to output folder. If None, replaces original files

    Returns:
        list: List of paths to processed audio files
    """
    # Supported audio file extensions
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

    # Create output folder if specified
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    processed_files = []

    # Process each audio file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip if not a file or not an audio file
        if not os.path.isfile(file_path):
            continue

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in audio_extensions:
            continue

        # Generate output path
        base_name = os.path.splitext(filename)[0]

        if output_folder is not None:
            # Save to specified output folder
            output_path = os.path.join(output_folder, f"{base_name}_no_silence.wav")
        else:
            # Create temporary file to process, then replace original
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"temp_{filename}")

        try:
            print(f"Processing: {filename}")
            processed_path = remove_silence_from_file(file_path, output_path)

            if output_folder is None:
                # Replace original file with processed version
                shutil.move(processed_path, file_path)
                processed_files.append(file_path)
                print(f"Replaced original: {file_path}")
            else:
                processed_files.append(processed_path)
                print(f"Saved: {processed_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    if output_folder is None:
        print(f"\nProcessed and replaced {len(processed_files)} files successfully.")
    else:
        print(f"\nProcessed {len(processed_files)} files successfully.")
    return processed_files


# Example usage for the original single file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove silence from audio files or folders"
    )
    parser.add_argument(
        "path", help="Path to audio file or folder containing audio files"
    )
    parser.add_argument(
        "-o", "--output", help="Output folder for processed files (optional)"
    )

    args = parser.parse_args()

    if os.path.isfile(args.path):
        # Process single file
        if args.output:
            output_path = args.output
        else:
            # Create temporary file to process, then replace original
            temp_dir = tempfile.gettempdir()
            base_name = os.path.basename(args.path)
            temp_output = os.path.join(temp_dir, f"temp_{base_name}")
            output_path = temp_output

        try:
            processed_path = remove_silence_from_file(args.path, output_path)

            if not args.output:
                # Replace original file with processed version
                shutil.move(processed_path, args.path)
                print(f"Processed and replaced original file: {args.path}")
            else:
                print(f"Processed single file: {processed_path}")
        except Exception as e:
            print(f"Error processing file: {str(e)}")

    elif os.path.isdir(args.path):
        # Process folder
        try:
            processed_files = process_folder(args.path, args.output)
            print(f"Processed {len(processed_files)} files successfully.")
        except Exception as e:
            print(f"Error processing folder: {str(e)}")

    else:
        print(f"Error: '{args.path}' is not a valid file or directory.")
