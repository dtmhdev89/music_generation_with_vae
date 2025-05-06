import json
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class AudioDatasetPreprocess:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_and_get_genres(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        return data.get('genres', [])

    @staticmethod
    def load_and_resample_audio(file_path, target_sr=22050):
        audio, sr = librosa.load(file_path, sr=None)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        return audio, target_sr

    @staticmethod
    def audio_to_melspec(
        audio,
        sr,
        n_mels,
        n_fft=2048,
        hop_length=512,
        to_db=False
    ):
        spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_mels=n_mels
        )

        if to_db:
            spec = librosa.power_to_db(spec, ref=np.max)

        return spec
    
    @staticmethod
    def normalize_melspec(melspec, norm_range=(0, 1)):
        scaler = MinMaxScaler(feature_range=norm_range)
        melspec = melspec.T
        melspec_normalized = scaler.fit_transform(melspec)

        return melspec_normalized.T
    
    @staticmethod
    def denormalize_melspec(
        melspec_normalized,
        original_melspec,
        norm_range=(0, 1)
    ):
        scaler = MinMaxScaler(feature_range=norm_range)
        melspec = original_melspec.T
        scaler.fit(melspec)
        melspec_denormalized = scaler.inverse_transform(melspec_normalized.T)

        return melspec_denormalized.T

    @staticmethod
    def melspec_to_audio(melspec, sr, n_fft=2048, hop_length=512, n_iter=64):
        if np.any(melspec < 0):
            melspec = librosa.db_to_power(melspec)

        audio_reconstructed = librosa.feature.inverse.mel_to_audio(
            melspec,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=n_iter
        )
        
        return audio_reconstructed
    
    @staticmethod
    def display_audio_files(
        reconstructed_audio,
        sr, title="", original_audio=None):
        if original_audio is not None:
            print("Original Audio:")
            ipd.display(ipd.Audio(original_audio, rate=sr))
            print("Reconstructed Audio (from Mel Spectrogram):")
        else:
            print(title)

        ipd.display(ipd.Audio(reconstructed_audio, rate=sr))

    @staticmethod
    def show_spectrogram(
        spectrogram,
        title="Mel-Spectrogram",
        denormalize=False,
        is_numpy=False
    ):
        if not is_numpy:
            spectrogram = spectrogram.squeeze().cpu().numpy()
        plt.figure(figsize=(10, 4))
        if denormalize:
            plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
        else:
            plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency")
        plt.colorbar()
        plt.show()
