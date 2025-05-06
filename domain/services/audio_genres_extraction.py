import os
import json
from music_generation_with_vae.configs.constant import Constant
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
                AudioGenresExtraction.load_data(unique_genres_path)
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
            AudioGenresExtraction.save_data(unique_genres, unique_genres_path)
        
        max_genres = len(unique_genres)

        return unique_genres, max_genres

    @staticmethod
    def preload_data(file_path):
        if os.path.exists(file_path):
            return AudioGenresExtraction.load_data(file_path)
        
        return None

    @staticmethod
    def save_data(data, save_file_path):
        try:
            with open(save_file_path, "w") as f:
                if isinstance(data, set):
                    data = list(data)

                json.dump(data, f, indent=4)
            print(f"Data successfully saved to {save_file_path}")
        except Exception as e:
            print(f"Error saving data to JSON file: {e}")

    @staticmethod
    def load_data(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            print(f"Data successfully loaded from {file_path}")

            return data
        except Exception as e:
            print(f"Error loading data from JSON file: {e}")
            return None
