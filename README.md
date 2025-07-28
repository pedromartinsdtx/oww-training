# Requirements
```sh
sudo apt install -y portaudio19-dev pulseaudio pulseaudio-utils ffmpeg
```

# Useful commands
```sh
# List available audio devices
pactl list short sources
# Play an audio file
paplay <audio_file>
```


## 🎯 Goal (Restated):

* **Word:** "Clarisse"
* **Language:** Portuguese
* **Variability:** Emotion, intonation, rhythm, or tone changes — not context phrases
* **Use Case:** Train a **wakeword detection model** with diverse synthetic samples

---

## 🧠 Strategy: Prompt Template for Voice LLM / TTS

If you're using a **prompt-to-audio model** (like Bark, Tortoise, or ElevenLabs), you **control speech variations via the prompt**, even if the text is just "Clarisse".

You vary:

* Descriptive context: Emotion, tone, pace, accent
* Explicit instruction (if LLM-based): e.g., “Say ‘Clarisse’ slowly, in a happy tone, in Brazilian Portuguese.”

---

## 🧩 Prompt Template Examples (Portuguese-Focused)

```python
import random

tones = [
    "em tom calmo",
    "em tom animado",
    "em tom sério",
    "em voz sussurrada",
    "em voz firme",
    "em voz suave",
    "em voz entusiasmada",
    "em voz cansada",
    "com sotaque brasileiro",
    "com sotaque de Portugal",
    "com emoção",
    "como se chamasse alguém",
    "com leve sorriso",
]

contexts = [
    "Diga apenas a palavra 'Clarisse'",
    "Fale 'Clarisse'",
    "Pronuncie 'Clarisse'",
    "Repita 'Clarisse'",
]

def generate_prompt():
    context = random.choice(contexts)
    tone = random.choice(tones)
    return f"{context}, {tone}."

# Example
for _ in range(5):
    print(generate_prompt())
```

---

### 💬 Sample Output:

```
Fale 'Clarisse', em tom animado.
Diga apenas a palavra 'Clarisse', com sotaque brasileiro.
Pronuncie 'Clarisse', em voz suave.
Repita 'Clarisse', como se chamasse alguém.
Diga apenas a palavra 'Clarisse', em tom sério.
```

---

## 🧪 How to Use With Voice Generation

If you're using a model like **Bark** or **ElevenLabs**, you can pass the generated prompt directly to the TTS function:

```python
# Pseudo-code using a TTS API (replace with your actual TTS)
audio = tts_generate(prompt_text=generate_prompt(), language="pt")
```

This would produce audio that **says only "Clarisse"**, but in different expressive ways.

---

## ✅ Summary

* You vary the **how** (tone, emotion), not the **what** ("Clarisse").
* The output is always a natural variation of **just the word "Clarisse"**, spoken in **Portuguese**.
* Use prompt engineering to guide the audio model to pronounce it differently.

