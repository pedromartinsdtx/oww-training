import base64
import os
import random
import uuid
from typing import Optional

from openai import AsyncOpenAI, OpenAI
from openai.helpers import LocalAudioPlayer

#! NÃO FUNCIONA NEM COM UMA API_KEY DO JACINTO, PORQUE É PRECISO PAGAR NA MESMA.

openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input = """Clarisse"""

instructions = """Responde apenas com a palavra \"Clarisse\", em português de Portugal, como se estivesses a chamar um assistente virtual — no estilo de dizer \"Alexa\" ou \"Siri\"."""

OPENAIFM_AVAILABLE_VOICES = [
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

BASE_OUTPUT_DIR = "samples/openai-fm/clrs-openai"


async def generate_openai_voice(
    voice: Optional[str] = None,
) -> None:
    # Arguments handling
    if voice is None:
        voice = random.choice(OPENAIFM_AVAILABLE_VOICES)

    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=input,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)


def generate_openai_voice_sync(
    voice: Optional[str] = None,
) -> None:
    # Arguments handling
    if voice is None:
        voice = random.choice(OPENAIFM_AVAILABLE_VOICES)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
        messages=[
            {"role": "user", "content": instructions},
        ],
    )

    print(completion.choices[0])

    id = str(uuid.uuid4())

    wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
    with open(f"{BASE_OUTPUT_DIR}{id}.wav", "wb") as f:
        f.write(wav_bytes)


if __name__ == "__main__":
    import asyncio

    asyncio.run(generate_openai_voice())
    # generate_openai_voice_sync()
