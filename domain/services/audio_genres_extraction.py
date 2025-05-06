import os
import json
from music_generation_with_vae.configs.constant import Constant
from music_generation_with_vae.utils.file_utils import FileUtils
from music_generation_with_vae.domain.services.audio_dataset_preprocess import AudioDatasetPreprocess


class AudioGenresExtraction:
    """Audio Gernes Extraction Class"""
    def __init__(self) -> None:
        self.__all_genres: list = []
        self.__json_dir = Constant.PIANO_AUDIO_JSON_PATH

    @property
    def json_dir(self):
        """__json_dir getter"""

        return self.__json_dir

    def extract(self, force=False):
        """Do Audio data gernes extraction"""
        unique_genres_path = os.path.join(
            Constant.PRELOAD_DATA_PATH,
            "unique_genres.json"
        )

        if (not force) and os.path.exists(unique_genres_path):
            unique_genres = set(
                FileUtils.load_data(unique_genres_path)
            )
        else:
            for filename in os.listdir(self.json_dir):
                if filename.endswith('.json'):
                    json_path = os.path.join(self.json_dir, filename)
                    genres = AudioDatasetPreprocess.load_and_get_genres(
                        json_path
                    )
                    self.__all_genres.extend(genres)

            unique_genres = set(self.__all_genres)
            FileUtils.save_data(unique_genres, unique_genres_path)
        
        max_genres = len(unique_genres)

        return unique_genres, max_genres

    @staticmethod
    def preload_data(file_path):
        if os.path.exists(file_path):
            return FileUtils.load_data(file_path)

        return None
