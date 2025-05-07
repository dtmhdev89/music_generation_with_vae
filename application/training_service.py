import os
import torch
import torch.nn as nn
import librosa.display
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from music_generation_with_vae.configs.constant import Constant
from music_generation_with_vae.domain.services.audio_dataset_preprocess import AudioDatasetPreprocess
from music_generation_with_vae.domain.services.audio_genres_extraction import AudioGenresExtraction
from music_generation_with_vae.domain.models.audio_dataset import AudioDataset
from music_generation_with_vae.domain.services.audio_tokenizer import AudioTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingService(nn.Module):
    """Traning Service Class"""

    def __init__(self):
        super(TrainingService, self).__init__()

        self._unique_genres, \
            self._max_genres = AudioGenresExtraction().extract()
        self._audio_tokenizer = AudioTokenizer(self._unique_genres)
        self._audio_dir = os.path.join(Constant.PIANO_AUDIO_JSON_PATH, "audio")
        self._json_dir = Constant.PIANO_AUDIO_JSON_PATH
        self._sample_rate = 22050
        self._duration = 3
        self._n_mels = 256
        self._testset_amount = 32
        self._trainset, self._testset = self._split_train_test_set()
    
    @property
    def n_mels(self):
        """_n_mels getter"""

        return self._n_mels
    
    @property
    def max_genres(self):
        """_max_genres getter"""

        return self._max_genres
    
    @property
    def audio_tokenizer(self):
        """_audio_tokenizer getter"""

        return self._audio_tokenizer

    def _split_train_test_set(self):
        """Split train, test set from dataset"""

        trainset = AudioDataset(
            data_dir=self._audio_dir,
            json_dir=self._json_dir,
            sample_rate=self._sample_rate,
            duration=self._duration,
            n_mels=self._n_mels,
            n_genres=self._max_genres,
            testset_amount=self._testset_amount,
            audio_tokenizer=self._audio_tokenizer
        )

        if len(trainset) == 0:
            raise ValueError(f"No .wav file found in {self._audio_dir}.")
        
        testset = trainset.testset

        return trainset, testset

    def make_train_test_loader(self):
        """Make train, test loader"""

        trainloader = DataLoader(
            self._trainset,
            batch_size=128,
            shuffle=True,
            num_workers=4
        )
        testloader = DataLoader(
            self._testset,
            batch_size=self._testset_amount,
            shuffle=False,
            num_workers=4
        )

        return trainloader, testloader

    def display_demo_mel_spectrogram(self):
        trainloader, _ = self.make_train_test_loader()
        audios = trainloader.dataset.audios.copy()

        frame = audios[0][0].shape[-1]
        print(audios[0][1])
        print(frame)

        for i in range(1):
            AudioDatasetPreprocess.show_spectrogram(
                audios[i][0],
                title="Clip Mel-Spectrogram"
            )
            spec_denorm = AudioDatasetPreprocess.denormalize_melspec(
                audios[i][0].numpy().squeeze(),
                audios[i][2].numpy().squeeze()
            )
            AudioDatasetPreprocess.show_spectrogram(
                torch.tensor(spec_denorm),
                title="Denormalized Mel-Spectrogram",
                denormalize=True
            )
            audio_reconstructed = AudioDatasetPreprocess.melspec_to_audio(
                spec_denorm,
                self._sample_rate
            )
            AudioDatasetPreprocess.display_audio_files(
                audio_reconstructed,
                self._sample_rate,
                title="Original Audio after convert to Spectrogram and back to Audio"
            )

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        """Loss Function"""

        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + KLD

    def train_vae(
        self,
        model,
        dataloader,
        optimizer,
        scheduler,
        num_epochs,
        verbose_interval=50
    ):
        """Training CVAE model"""

        model.train()
        losses = []
        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            train_loss = 0
            for batch_idx, (data, genres_input, ori_data) in enumerate(dataloader):
                data = data.to(device)
                genres_input = genres_input.to(device)

                optimizer.zero_grad()

                recon, mu, logvar = model(data, genres_input)
                loss = self.__class__.loss_function(recon, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            scheduler.step()
            avg_loss = train_loss / len(dataloader.dataset)
            losses.append(avg_loss)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Lr: {scheduler.get_last_lr()[0]}")

            if epoch == 0 or (epoch + 1) % verbose_interval == 0:
                data = data[0].detach().cpu()
                recon_img = recon[0].detach().cpu()
                AudioDatasetPreprocess.show_spectrogram(
                    data,
                    title="Original Spectrogram"
                )
                AudioDatasetPreprocess.show_spectrogram(
                    recon_img,
                    title="Reconstructed Spectrogram"
                )

        return mu, logvar, losses

    def plot_losses(
        self,
        losses,
        title="Training Loss",
        xlabel="Epochs",
        ylabel="Loss",
        color='b',
        grid=True
    ):
        """Plot losses"""

        plt.figure(figsize=(10, 6))
        plt.plot(losses, color=color, linewidth=2)
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(grid, linestyle="--", alpha=0.6)

        min_loss_idx = losses.index(min(losses))
        max_loss_idx = losses.index(max(losses))

        plt.annotate(
            f"Min Loss: {min(losses):.4f}",
            xy=(min_loss_idx, min(losses)),
            xytext=(min_loss_idx + 1, min(losses) + 0.1),
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=12, color='green'
        )

        plt.annotate(
            f"Max Loss: {max(losses):.4f}",
            xy=(max_loss_idx, max(losses)),
            xytext=(max_loss_idx + 1, max(losses) + 0.1),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=12, color="red"
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def write_losses_to_file(
        losses,
        log_dir="logs",
        filename="training_losses.txt"
    ):
        """
        Writes a list of training losses to a text file, creating the directory if it doesn't exist.

        Args:
            losses (list): A list of loss values.  It is assumed that the index
                        of the loss corresponds to the epoch number (starting from 0).
            log_dir (str, optional): The name of the directory to store the log file.
                Defaults to "logs".
            filename (str, optional): The name of the text file.
                Defaults to "training_losses.txt".
        """
        # Create the log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        filepath = os.path.join(log_dir, filename)

        try:
            with open(filepath, "w") as f:
                for epoch, loss in enumerate(losses):
                    f.write(f"{epoch}, {loss}\n")  # Write each epoch and loss on a new line
            print(f"Losses successfully written to {filepath}")
        except Exception as e:
            print(f"Error writing losses to file: {e}")
