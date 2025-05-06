import json
import os
import torch


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

    @staticmethod
    def save_tensor_data(data, save_file_path):
        """Save data in Tensor format"""

        try:
            _, file_ext = os.path.splitext(save_file_path)
            if file_ext.lower() != "pt":
                raise ValueError(f"File {save_file_path} is not pt")

            torch.save(data, save_file_path)
            print(f"Data successfully saved to {save_file_path}")
        except Exception as e:
            print(f"Error saving data to Tensor file: {e}")

    @staticmethod
    def load_tensor_data(file_path, map_location=None):
        """Load tensor file"""

        try:
            if map_location:
                return torch.load(file_path, map_location=map_location)
            else:
                return torch.load(file_path)
        except Exception as e:
            print(f"Error to load Tensor file: {e}")

