from typing import List


class Configurations:
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        gamma: float,
        n_mels: int,
        n_genres: int,
        lr: float | None = 2e-4,
        num_epochs: int | None = 100
    ) -> None:
        self.__d_model = d_model
        self.__latent_dim = latent_dim
        self.__gamma = gamma
        self.__lr = lr
        self.__num_epochs = num_epochs
        self.__step_size = self.__num_epochs // 2
        self.__verbose_interval = self.__num_epochs // 10
        self.__n_mels = n_mels
        self.__n_genres = n_genres

    def __key_list(self) -> List[str]:
        return [
            "d_model",
            "latent_dim",
            "gamma",
            "lr",
            "num_epochs",
            "step_size",
            "verbose_interval",
            "n_mels",
            "n_genres"
        ]

    @property
    def to_dict(self):
        return {
            k: getattr(self, f"__{k}")
            for k in self.__key_list()
        }
