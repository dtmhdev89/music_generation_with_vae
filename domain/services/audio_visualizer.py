import os

from music_generation_with_vae.domain.services.audio_dataset_preprocess import AudioDatasetPreprocess
from music_generation_with_vae.configs.constant import Constant


class AudioVisualizer:

    @staticmethod
    def visualize(audio_name):
        audio, target_sr = AudioDatasetPreprocess.load_and_resample_audio(
            os.path.join(Constant.PIANO_AUDIO_JSON_PATH, "audio", audio_name)
        )

        mel_spectrogram = AudioDatasetPreprocess.audio_to_melspec(
            audio,
            target_sr,
            n_mels=256
        )

        AudioDatasetPreprocess.show_spectrogram(
            mel_spectrogram,
            title="Original Mel-Spectrogram",
            is_numpy=True
        )
