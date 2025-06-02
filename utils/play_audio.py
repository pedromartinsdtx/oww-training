import subprocess


def play_audio_file(file_path):
    """Play audio file using paplay."""
    try:
        subprocess.run(["paplay", file_path], check=True, stderr=subprocess.DEVNULL)
        print("Audio played successfully")
    except subprocess.CalledProcessError:
        print(f"Could not play audio file: {file_path}")
        print("Ensure PulseAudio is running and audio hardware is available.")
    except FileNotFoundError:
        print("paplay not found. Install PulseAudio: sudo apt install pulseaudio-utils")
        print(f"Audio file available at: {file_path}")


if __name__ == "__main__":
    test_file = "samples/clarisse_gem_Charon_0_5a95953a-4741-4499-b3bd-3fb90ef65c9c.wav"
    play_audio_file(test_file)
