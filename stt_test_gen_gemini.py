import io
import mimetypes
import os
import random
import struct
import tempfile
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- BEGIN: Copied/Adapted from gen_gemini.py ---
load_dotenv()

GEMINI_AVAILABLE_VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]

GEMINI_AVAILABLE_MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]

MAX_REQUESTS_PER_MINUTE = 3
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE  # = 20 seconds


def _resample_audio(
    audio_data: np.ndarray, original_samplerate: int, target_samplerate: int
) -> np.ndarray:
    if audio_data.ndim > 1 and audio_data.shape[1] == 1:
        audio_data = audio_data[:, 0]
    resampled_audio = librosa.resample(
        y=audio_data, orig_sr=original_samplerate, target_sr=target_samplerate
    )
    return resampled_audio


def convert_bytes_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}


def save_binary_file(file_name, data):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")


def generate_gemini_voice(
    prompt: str,
    output_filename: str,
    model: Optional[str] = "gemini-2.5-flash-preview-tts",
    voice: Optional[str] = None,
    temperature: Optional[float] = 1.0,
    resample_to: Optional[int] = 16000,
) -> str:
    if voice is None:
        voice = random.choice(GEMINI_AVAILABLE_VOICES)
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if (
            chunk.candidates[0].content.parts[0].inline_data
            and chunk.candidates[0].content.parts[0].inline_data.data
        ):
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                file_extension = ".wav"
                data_buffer = convert_bytes_to_wav(
                    inline_data.data, inline_data.mime_type
                )
            # Resample audio if needed
            audio_np, orig_sr = sf.read(io.BytesIO(data_buffer), dtype="float32")
            if audio_np.ndim > 1 and audio_np.shape[1] == 1:
                audio_np = audio_np[:, 0]
            if resample_to and orig_sr != resample_to:
                audio_np = _resample_audio(audio_np, orig_sr, resample_to)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
                    sf.write(tmpf.name, audio_np, resample_to)
                    tmpf.seek(0)
                    data_buffer = tmpf.read()
                file_extension = ".wav"
            save_binary_file(output_filename, data_buffer)
            return output_filename
        else:
            print(f"No inline data found in chunk with prompt '{prompt}'.")
            print(chunk.text)
    return None


# --- END: Copied/Adapted from gen_gemini.py ---

# --- BEGIN: Test items (replace with your own list or load from file) ---
test_items = [
    # {
    #     "test_id": "17_clarisse_liga_a_televisao",
    #     "transcription": "Clarisse, liga a televisão da sala.",
    #     "filename": "17_clarisse_liga_a_televisao.wav",
    # },
    # {
    #     "test_id": "18_clarisse_define_um_alarme",
    #     "transcription": "Clarisse, define um alarme para as sete da manhã.",
    #     "filename": "18_clarisse_define_um_alarme.wav",
    # },
    # {
    #     "test_id": "19_qual_e_a_temperatura_em_lisboa",
    #     "transcription": "Qual é a temperatura em Lisboa agora?",
    #     "filename": "19_qual_e_a_temperatura_em_lisboa.wav",
    # },
    # {
    #     "test_id": "20_clarisse_desliga_a_tomada_do_quarto",
    #     "transcription": "Clarisse, desliga a tomada do quarto.",
    #     "filename": "20_clarisse_desliga_a_tomada_do_quarto.wav",
    # },
    # {
    #     "test_id": "21_clarisse_adiciona_leite_a_lista_de_compras",
    #     "transcription": "Clarisse, adiciona leite à lista de compras.",
    #     "filename": "21_clarisse_adiciona_leite_a_lista_de_compras.wav",
    # },
    # {
    #     "test_id": "22_clarisse_que_horas_sao",
    #     "transcription": "Clarisse, que horas são?",
    #     "filename": "22_clarisse_que_horas_sao.wav",
    # },
    # {
    #     "test_id": "23_clarisse_vai_chover_amanha",
    #     "transcription": "Clarisse, vai chover amanhã?",
    #     "filename": "23_clarisse_vai_chover_amanha.wav",
    # },
    # {
    #     "test_id": "24_clarisse_conta_me_uma_piada",
    #     "transcription": "Clarisse, conta-me uma piada.",
    #     "filename": "24_clarisse_conta_me_uma_piada.wav",
    # },
    # {
    #     "test_id": "25_clarisse_apaga_luzes_e_fecha_estores",
    #     "transcription": "Clarisse, apaga as luzes da sala e fecha os estores, por favor.",
    #     "filename": "25_clarisse_apaga_luzes_e_fecha_estores.wav",
    # },
    # {
    #     "test_id": "26_clarisse_lembra_me_de_comprar_pao_amanha_de_manha",
    #     "transcription": "Clarisse, lembra-me de comprar pão amanhã de manhã quando eu sair de casa.",
    #     "filename": "26_clarisse_lembra_me_de_comprar_pao_amanha_de_manha.wav",
    # },
    # {
    #     "test_id": "27_clarisse_qual_e_a_previsao_para_o_fim_de_semana_em_porto",
    #     "transcription": "Clarisse, qual é a previsão do tempo para o fim de semana no Porto?",
    #     "filename": "27_clarisse_qual_e_a_previsao_para_o_fim_de_semana_em_porto.wav",
    # },
    # {
    #     "test_id": "28_clarisse_podes_ajustar_o_brilho_para_50_porcento",
    #     "transcription": "Clarisse, podes ajustar o brilho da luz da cozinha para cinquenta por cento?",
    #     "filename": "28_clarisse_podes_ajustar_o_brilho_para_50_porcento.wav",
    # },
    # {
    #     "test_id": "29_clarisse_que_reminders_tenho_para_hoje",
    #     "transcription": "Clarisse, que lembretes tenho marcados para hoje?",
    #     "filename": "29_clarisse_que_reminders_tenho_para_hoje.wav",
    # },
    # {
    #     "test_id": "30_clarisse_se_chover_lembra_me_do_guarda_chuva",
    #     "transcription": "Clarisse, se chover amanhã lembra-me de levar o guarda-chuva.",
    #     "filename": "30_clarisse_se_chover_lembra_me_do_guarda_chuva.wav",
    # },
    # {
    #     "test_id": "31_podes_diminuir_a_luz_do_quarto",
    #     "transcription": "Podes diminuir a luz do quarto para vinte por cento?",
    #     "filename": "31_podes_diminuir_a_luz_do_quarto.wav",
    # },
    # {
    #     "test_id": "32_lembra_me_de_ir_ao_supermercado_hoje_a_tarde",
    #     "transcription": "Lembra-me de ir ao supermercado hoje à tarde depois do trabalho.",
    #     "filename": "32_lembra_me_de_ir_ao_supermercado_hoje_a_tarde.wav",
    # },
    # {
    #     "test_id": "33_esta_frio_la_fora",
    #     "transcription": "Está frio lá fora?",
    #     "filename": "33_esta_frio_la_fora.wav",
    # },
    # {
    #     "test_id": "34_que_dispositivos_estao_ligados_na_cozinha",
    #     "transcription": "Que dispositivos estão ligados na cozinha neste momento?",
    #     "filename": "34_que_dispositivos_estao_ligados_na_cozinha.wav",
    # },
    # {
    #     "test_id": "35_define_um_timer_para_o_forno",
    #     "transcription": "Define um timer para o forno de quarenta minutos, por favor.",
    #     "filename": "35_define_um_timer_para_o_forno.wav",
    # },
    # {
    #     "test_id": "36_amanha_vou_precisar_de_uma_sombrinha",
    #     "transcription": "Amanhã vou precisar de uma sombrinha?",
    #     "filename": "36_amanha_vou_precisar_de_uma_sombrinha.wav",
    # },
    # {
    #     "test_id": "37_pedir_informacao_comboio",
    #     "transcription": "Gostava de saber a que horas parte o próximo comboio para o Porto e se faz paragem em Coimbra, porque preciso mesmo de chegar cedo amanhã.",
    #     "filename": "37_pedir_informacao_comboio.wav",
    # },
    # {
    #     "test_id": "38_pedir_cafe_pastelaria",
    #     "transcription": "Se puderes, traz-me um café cheio e um pastel de nata bem quentinho, daqueles que acabaram de sair do forno, como só se faz cá em Lisboa.",
    #     "filename": "38_pedir_cafe_pastelaria.wav",
    # },
    # {
    #     "test_id": "39_conversa_sobre_tempo",
    #     "transcription": "Hoje está um daqueles dias em que o céu ameaça chover, mas no fim acaba por não cair nem uma pinga. Isto é mesmo típico do tempo em Abril.",
    #     "filename": "39_conversa_sobre_tempo.wav",
    # },
    # {
    #     "test_id": "40_pedir_direcoes_rua",
    #     "transcription": "Desculpe, pode dizer-me como chego à Rua dos Sapateiros? Sei que fica ali para os lados da Baixa, mas perco-me sempre nestas ruelas estreitas.",
    #     "filename": "40_pedir_direcoes_rua.wav",
    # },
    # {
    #     "test_id": "41_falar_sobre_futebol",
    #     "transcription": "Ontem o jogo foi renhido até ao fim, mas aquele golo aos noventa minutos deixou toda a gente de boca aberta. Só mesmo em Portugal é que se vive o futebol assim!",
    #     "filename": "41_falar_sobre_futebol.wav",
    # },
]
# --- END: Test items ---


def main():
    output_dir = (
        Path.home() / "Clarisse_Chatbot/tests/speech_to_text/input_tests/audios"
    )
    # output_dir = "samples/gemini_batch"
    os.makedirs(output_dir, exist_ok=True)
    for idx, item in enumerate(test_items):
        prompt = f"Diz apenas uma vez rapidamente a seguinte frase em português de Portugal: {item['transcription']}"
        output_filename = os.path.join(output_dir, item["filename"])
        print(f"[{idx + 1}/{len(test_items)}] Generating: {output_filename}")
        try:
            generate_gemini_voice(prompt, output_filename)
        except Exception as e:
            print(f"Error generating {output_filename}: {e}")
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"Rate limit hit. Waiting {int(REQUEST_INTERVAL) + 5} seconds...")
                time.sleep(int(REQUEST_INTERVAL) + 5)
            else:
                time.sleep(5)
        print(f"Waiting {int(REQUEST_INTERVAL)} seconds before next request...")
        time.sleep(REQUEST_INTERVAL)


if __name__ == "__main__":
    main()
