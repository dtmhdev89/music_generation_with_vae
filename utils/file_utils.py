import json
import os


class FileUtils:
    """File Utils"""

    @staticmethod
    def save_data(data, save_file_path):
        """Save data in json format"""

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
        """Load data from json format"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            print(f"Data successfully loaded from {file_path}")

            return data
        except Exception as e:
            print(f"Error loading data from JSON file: {e}")
            return None
