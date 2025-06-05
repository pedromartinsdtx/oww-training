import threading

from gen_edge import generate_edge_tts_voice_loop
from gen_gemini import gemini_loop


def main():
    threads = [
        threading.Thread(target=gemini_loop, args=(15,)),
        threading.Thread(target=generate_edge_tts_voice_loop),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
