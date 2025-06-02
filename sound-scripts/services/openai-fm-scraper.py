import random
import time
import uuid
import requests

from utils.play_audio import play_audio_file

#! Thhs script makes you get banned from OpenAI-FM if you run it too many times.

OPENAI_FM_AVAILABLE_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
]

BASE_OUTPUT_DIR = "samples/openai-fm/clrs-openai-fm"


def generate_and_save_audio(
    input_text, instructions_text, voice="shimmer", vibe="null", filename="clarisse.mp3"
):
    url = "https://www.openai.fm/api/generate"
    headers = {
        "accept": "*/*",
        "accept-language": "pt-PT,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "priority": "u=1, i",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "Referer": "https://www.openai.fm/",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }

    files = {
        "input": (None, input_text),
        "prompt": (None, instructions_text),
        "voice": (None, voice),
        "vibe": (None, vibe),
    }

    print("Sending voice generation request...")
    response = requests.post(url, headers=headers, files=files)

    if response.status_code != 200:
        raise Exception(f"POST failed: {response.status_code} — {response.text}")

    # Save binary audio directly
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"✅ Audio saved as '{filename}'")


def generate_openai_voice():
    input_text = "Clarisse"
    instructions_text = (
        "Diz apenas a palavra 'Clarisse', em português de Portugal, "
        "como se estivesses a chamar um assistente virtual — no estilo de dizer 'Alexa' ou 'Siri'."
    )
    voice = random.choice(OPENAI_FM_AVAILABLE_VOICES)

    id = str(uuid.uuid4())

    filename = f"{BASE_OUTPUT_DIR}-{voice}-{id}.mp3"
    try:
        generate_and_save_audio(
            input_text, instructions_text, voice=voice, filename=filename
        )
        play_audio_file(filename)
    except Exception as e:
        print(f"Error generating audio for '{filename}': {e}")
        raise e


def generate_multiple_openai_voices(
    num_voices: int = 20,
) -> None:
    for i in range(num_voices):
        print(f"Voice Gen OpenAI-FM: (Attempt {i + 1})")
        try:
            generate_openai_voice()
            print(f"Voice Gen OpenAI-FM: {i + 1} successful.")

            sleep = random.randint(5, 10)
            print(f"Waiting {sleep} seconds for the next request...")
            time.sleep(sleep)

        except Exception as e:
            error_str = str(e)
            if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in error_str:
                print("Daily quota limit reached. Halting further requests.")
                break
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(error_str)
                sleep = 40
                print(
                    f"Rate limit hit. Waiting {sleep} seconds for the next request..."
                )
                time.sleep(sleep)
            else:
                print(f"Voice Gen OpenAI-FM: ERROR-(Attempt {i + 1}):\n{e}")
                time.sleep(5)


if __name__ == "__main__":
    generate_multiple_openai_voices()
