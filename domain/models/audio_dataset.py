import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from music_generation_with_vae.utils.file_utils import FileUtils
from music_generation_with_vae.domain.services.audio_dataset_preprocess import AudioDatasetPreprocess
from music_generation_with_vae.domain.services.audio_tokenizer import AudioTokenizer
from music_generation_with_vae.configs.constant import Constant


class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        json_dir,
        sample_rate,
        duration,
        n_mels,
        n_genres,
        audio_tokenizer: AudioTokenizer,
        testset_amount=10,
        force_audio_process=False
    ):
        self.data_dir = data_dir
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".mp3")
        ]
        self.json_dir = json_dir
        self.json_files = [
            os.path.join(json_dir, f)
            for f in os.listdir(json_dir) if f.endswith(".json")
        ]
        self.sample_rate = sample_rate
        self.duration = duration
        self.fixed_length = sample_rate * duration
        self.n_genres = n_genres
        self.n_mels = n_mels
        self._audio_tokenizer = audio_tokenizer

        audios_preload_path = os.path.join(
            Constant.PRELOAD_DATA_PATH,
            "transformed_audios.json"
        )

        if (not force_audio_process) and (os.path.exists(audios_preload_path)):
            audios = FileUtils.load_data(audios_preload_path)
        else:
            audios = self._transform_audios()
            FileUtils.save_data(
                data=audios,
                save_file_path=audios_preload_path
            )

        self.audios = audios[:len(audios) - testset_amount]
        self.testset = audios[len(audios) - testset_amount:]

        print(f"Loaded {len(self.audios)} audio segments from {len(self.files)} files, each with shape: {self.audios[0][0].shape}, {self.audios[0][1].shape}, duration: {duration} seconds")
        print(f"Test set: {len(self.testset)} audio segments")

    def _transform_audios(self):
        """Transform audios"""

        audios = []

        for file_path, json_file_path in tqdm(
            zip(self.files, self.json_files),
            desc=f"Loading audio files in {self.data_dir}",
            unit="file",
            total=len(self.files)
        ):
            audio, sr = AudioDatasetPreprocess.load_and_resample_audio(
                file_path,
                target_sr=self.sample_rate
            )
            genres_list = AudioDatasetPreprocess.load_and_get_genres(
                json_file_path
            )

            genres_tokens = self._audio_tokenizer.tokenize(genres_list)
            genres_input = self._audio_tokenizer.onehot_encode(
                genres_tokens,
                self.n_genres
            )
            genres_input = torch.tensor(
                genres_input,
                dtype=torch.long
            ).unsqueeze(0)

            n_samples = len(audio)
            n_segments = n_samples // self.fixed_length

            for i in range(n_segments):
                start = i * self.fixed_length
                end = (i + 1) * self.fixed_length
                segment = audio[start:end]
                mel_spec = AudioDatasetPreprocess.audio_to_melspec(
                    segment,
                    sr,
                    self.n_mels,
                    to_db=True
                )
                mel_spec_norm = AudioDatasetPreprocess.normalize_melspec(
                    mel_spec
                )
                mel_spec = torch.tensor(
                    mel_spec,
                    dtype=torch.float32
                ).unsqueeze(0)
                mel_spec_norm = torch.tensor(
                    mel_spec_norm,
                    dtype=torch.float32
                ).unsqueeze(0)
                audios.append((mel_spec_norm, genres_input, mel_spec))

        return audios

    def __len__(self):
        """Dataset len"""

        return len(self.audios)

    def __getitem__(self, idx):
        """Item process"""

        mel_spec_part, genres_input, mel_spec = self.audios[idx]

        return mel_spec_part, genres_input, mel_spec
