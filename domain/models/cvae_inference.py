import numpy as np
import torch.nn as nn
import torch
from music_generation_with_vae.domain.models.cvae import CVAE
from music_generation_with_vae.domain.services.audio_dataset_preprocess import AudioDatasetPreprocess
from music_generation_with_vae.application.training_service import TrainingService


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CVAEInference(nn.Module):
    def __init__(
        self,
        weight_path,
        inference_configs,
        audio_tokenizer
    ) -> None:
        super(CVAEInference, self).__init__()

        self.__weight_path = weight_path
        self.__inference_configs = inference_configs
        self.__model = CVAE(
            d_model=inference_configs["d_model"],
            latent_dim=inference_configs["latent_dim"],
            n_frames=inference_configs["n_frames"],
            n_mels=inference_configs["n_mels"],
            n_genres=inference_configs["n_genres"]
        ).to(device)
        self.__model.load_state_dict(
            torch.load(weight_path),
            map_location=device
        )
        self.__model.eval()
        self._audio_tokenizer = audio_tokenizer

    @property
    def model(self):
        """__model getter"""

        return self.__model

    @property
    def inference_configs(self):
        """____inference_configs gettern"""

        return self.__inference_configs

    @property
    def weight_path(self):
        """weight_path getter"""

        return self.__weight_path

    def inference(self, testloader):
        """Model inference"""

        with torch.no_grad():
            data, genres_input, ori_data = next(iter(testloader))
            data = data.to(device)
            genres_input = genres_input.to(device)
            recon, _, _ = self.model(data, genres_input)

            return recon, genres_input, ori_data
        
    def generate(
        self,
        dataloader,
        genres_list,
        num_samples=5,
        diff_level=1,
        sample_rate=22050
    ):
        with torch.no_grad():
            data, old_genres_input, ori_data = next(iter(dataloader))
            data = data.to(device)

            genres_tokens = self.audio_tokenizer.tokenize(genres_list)
            genres_input = self.audio_tokenizer.onehot_encode(genres_tokens, self.model.n_genres)
            genres_input = torch.tensor(
                genres_input, dtype=torch.long
            ).unsqueeze(0)
            genres_input = genres_input.repeat(old_genres_input.shape[0], 1)
            genres_input = genres_input.to(device)

            recon, mu, logvar = self.model(data, genres_input)
            ori_audios = []
            recon_audios = []
            for i in range(num_samples):
                old_genres_list = self._audio_tokenizer.detokenize_tolist(
                    self._audio_tokenizer.onehot_decode(
                        old_genres_input[i].squeeze().tolist()
                    )
                )
                AudioDatasetPreprocess.show_spectrogram(
                    data[i],
                    title=" ".join(
                       [
                            "Original Spectrogram with Genres:",
                            ", ".join(old_genres_list)
                       ]
                    )
                )
                AudioDatasetPreprocess.show_spectrogram(
                    recon[i],
                    title="Reconstructed Spectrogram with Genres: " + ", ".join(genres_list)
                )

                diff_spectrogram = torch.abs(data[i] - recon[i]) * diff_level
                AudioDatasetPreprocess.show_spectrogram(
                    diff_spectrogram,
                    title=f"Difference Spectrogram (|Original - Reconstructed|) * {diff_level}"
                )
                print(
                    "Loss: ",
                    TrainingService.loss_function(
                        recon[i],
                        data[i],
                        mu,
                        logvar
                    ).item()
                )

                spec_denorm = AudioDatasetPreprocess.denormalize_melspec(
                    recon[i].cpu().numpy().squeeze(),
                    ori_data[i].cpu().numpy().squeeze()
                )
                audio_reconstructed = AudioDatasetPreprocess.melspec_to_audio(
                    spec_denorm,
                    sr=sample_rate
                )
                ori_audio = AudioDatasetPreprocess.melspec_to_audio(
                    ori_data[i].cpu().numpy().squeeze(),
                    sr=sample_rate
                )

                recon_audios.append(audio_reconstructed)
                ori_audios.append(ori_audio)

                AudioDatasetPreprocess.display_audio_files(
                    ori_audio,
                    sample_rate,
                    title="Reconstructed Audio with Genres: " + ", ".join(old_genres_list)
                )
                AudioDatasetPreprocess.display_audio_files(
                    audio_reconstructed,
                    sample_rate,
                    title="Reconstructed Audio with Genres: " + ", ".join(genres_list)
                )

            if num_samples > 1:
                print("-"*100, "Connect all audio", "-"*100)
                recon_ori_audios = np.concatenate(ori_audios)
                AudioDatasetPreprocess.display_audio_files(
                    recon_ori_audios,
                    sample_rate, title="Connect all original audio"
                )
                recon_audios = np.concatenate(recon_audios)
                AudioDatasetPreprocess.display_audio_files(
                    recon_audios,
                    sample_rate, title="Connect all reconstructed audio"
                )
    