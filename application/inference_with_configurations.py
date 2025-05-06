from music_generation_with_vae.application.training_service import TrainingService
from music_generation_with_vae.domain.models.cvae_inference import CVAEInference
from music_generation_with_vae.domain.models.configurations import Configurations
import os
from dotenv import load_dotenv
load_dotenv(
    dotenv_path=os.environ.get(
        "ENV_FILE_PATH"
    )
)

if __name__ == "__main__":
    d_model = 64
    latent_dim = 128
    lr = 2e-4
    num_epochs = 100
    gamma = 0.5

    training_service = TrainingService()
    trainloader, testloader = training_service.make_train_test_loader()
    frame = trainloader.dataset.audios[0][0].shape[-1]

    inferent_configs = Configurations(
        d_model=d_model,
        latent_dim=latent_dim,
        lr=lr,
        num_epochs=num_epochs,
        gamma=gamma,
        n_mels=training_service.n_mels,
        n_genres=training_service.max_genres,
        n_frames=frame
    ).to_dict()

    audio_tokenizer = training_service.audio_tokenizer

    model = CVAEInference(
        weight_path=os.environ.get("WEIGHT_PATH"),
        inference_configs=inferent_configs,
        audio_tokenizer=audio_tokenizer
    )
    new_genres = ["Christmas", "Pop", "Halloween"]
    model.generate(testloader, new_genres, num_samples=2)
