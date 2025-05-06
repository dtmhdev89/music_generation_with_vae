from music_generation_with_vae.application.training_service import TrainingService
from music_generation_with_vae.domain.models.cvae import CVAE
from music_generation_with_vae.domain.models.configurations import Configurations
import torch.optim as optim
import torch
import time


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = 64
    latent_dim = 128
    lr = 2e-4
    num_epochs = 100
    gamma = 0.5

    training_service = TrainingService()
    trainloader, testloader = training_service.make_train_test_loader()
    audios = trainloader.dataset.audios.copy()
    frame = audios[0][0].shape[-1]

    train_configs = Configurations(
        d_model=d_model,
        latent_dim=latent_dim,
        lr=lr,
        num_epochs=num_epochs,
        gamma=gamma,
        n_mels=training_service.n_mels,
        n_genres=training_service.max_genres,
        n_frames=frame
    ).to_dict()

    model = CVAE(
        d_model=train_configs["d_model"],
        latent_dim=train_configs["latent_dim"],
        n_frames=train_configs["n_frames"],
        n_mels=train_configs["n_mels"],
        n_genres=train_configs["n_genres"]
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_configs["step_size"],
        gamma=train_configs["gamma"]
    )

    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    mu, logvar, losses = training_service.train_vae(
        model=model,
        dataloader=trainloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=train_configs["num_epochs"],
        verbose_interval=train_configs["verbose_interval"]
    )

    completed_trained_time = str(int(time.time()))

    torch.save(
        model.state_dict(),
        f"audio_vae_model_checkpoint_{completed_trained_time}.pth"
    )

    TrainingService.write_losses_to_file(
        losses=losses,
        filename=f"training_losses_{completed_trained_time}.txt"
    )

    training_service.plot_losses(
        losses=losses
    )
