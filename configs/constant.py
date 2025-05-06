import os
from dotenv import load_dotenv


load_dotenv(
    dotenv_path=os.environ.get(
        "ENV_FILE_PATH"
    )
)


class Constant:
    PIANO_AUDIO_PATH = os.environ.get(
        "PIANO_AUDIO_PATH",
        os.path.join("dataset/piano_audio_files")
    )
    PIANO_AUDIO_JSON_PATH = os.path.join(PIANO_AUDIO_PATH, "crawled_data")
    PRELOAD_DATA_PATH = os.environ.get(
        "PRELOAD_DATA_PATH",
        os.path.join("dataset/preload_data")
    )
