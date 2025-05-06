import os


class Constant:
    PIANO_AUDIO_PATH = os.environ.get(
        "PIANO_AUDIO_PATH",
        os.path.join("dataset/piano_audio_files")
    )
    PIANO_AUDIO_JSON_PATH = os.path.join(PIANO_AUDIO_PATH, "crawled_data")

