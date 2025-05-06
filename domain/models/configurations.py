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
        self._d_model = d_model
        self._latent_dim = latent_dim
        self._gamma = gamma
        self._lr = lr
        self._num_epochs = num_epochs
        self._step_size = self._num_epochs // 2
        self._verbose_interval = self._num_epochs // 10
        self._n_mels = n_mels
        self._n_genres = n_genres

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
            k: getattr(self, f"_{k}")
            for k in self.__key_list()
        }
