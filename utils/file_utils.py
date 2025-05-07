import json
import os
import torch
import gzip
import io
import h5py


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
    def save_tensor_data_with_gzip(data, save_file_path):
        """Save data in Tensor format
        This way costs more memory to perform since the Tensors is loaded in bytes to memory,
        and also gzip does it in memory too
        """

        try:
            _, file_ext = os.path.splitext(save_file_path)
            if file_ext.lower() != ".pt":
                raise ValueError(f"File {save_file_path} is not pt")

            with gzip.open(f"{save_file_path}.gz", "wb") as f:
                buffer = io.BytesIO()
                torch.save(data, buffer)
                buffer.seek(0)
                f.write(buffer.read())

            print(f"Data successfully saved to {save_file_path}")
        except Exception as e:
            print(f"Error saving data to Tensor file: {e}")

    @staticmethod
    def load_tensor_data_with_gzip(file_path, map_location=None):
        """Load tensor file"""

        try:
            with gzip.open(f"{file_path}.gz", "rb") as f:
                buffer = io.BytesIO(f.read())

                if map_location:
                    return torch.load(buffer, map_location=map_location)
                else:
                    return torch.load(buffer)
        except Exception as e:
            print(f"Error to load Tensor file: {e}")

    @staticmethod
    def save_tensor_data_with_hdf5(data, save_file_path):
        """Save Tensor data with HDF5
        This way converts the Tensor data into numpy and stream the data into disk to compress with gzip
        This will more efficient on memory
        """

        try:
            _, file_ext = os.path.splitext(save_file_path)
            if file_ext.lower() != ".h5":
                raise ValueError(f"File {save_file_path} is not .h5")

            with h5py.File(save_file_path, "w") as f:
                # save each tensor into a dataset in h5
                for i, tensor in enumerate(data):
                    f.create_dataset(
                        f"tensor_{i}",
                        data=tensor.numpy(),
                        compression="gzip"
                    )

            print(f"Data successfully saved to {save_file_path}")
        except Exception as e:
            print(f"Error saving data to Tensor file: {e}")

    @staticmethod
    def load_tensor_data_with_hdf5(file_path):
        """Load numpy data from h5 file and convert it into Tensor"""

        try:
            _, file_ext = os.path.splitext(file_path)
            if file_ext.lower() != ".h5":
                raise ValueError(f"File {file_path} is not .h5")
            
            loaded_tensors = []

            with h5py.File(file_path, "r") as f:
                for key in f.keys():
                    array = f[key][...]  # Load one at a time
                    loaded_tensors.append(torch.from_numpy(array))

            print(f"Data successfully loaded from {file_path}")

            return loaded_tensors
        except Exception as e:
            print(f"Error loading data to Tensor file: {e}")

