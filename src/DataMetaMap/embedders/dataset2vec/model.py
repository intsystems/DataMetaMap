from typing import Any, Type

import torch
from torch import Tensor, mean, nn, stack

from .config import Dataset2VecConfig, OptimizerConfig
from .train import LightningBase


class FeedForward(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        output_size: int,
        activation_cls: Type[nn.Module],
    ):
        super().__init__()
        assert n_layers >= 1, "Network must have at least one layer"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.activation_cls = activation_cls

        if n_layers == 1:
            self._init_single_layer()
        else:
            self._init_multiple_layers()

    def _init_single_layer(self) -> None:
        self.block = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            self.activation_cls()
        )

    def _init_multiple_layers(self) -> None:
        components = [
            nn.Linear(self.input_size, self.hidden_size),
            self.activation_cls(),
        ]
        for _ in range(self.n_layers - 2):
            components.append(nn.Linear(self.hidden_size, self.hidden_size))
            components.append(self.activation_cls())
        components.append(nn.Linear(self.hidden_size, self.output_size))
        components.append(self.activation_cls())
        self.block = nn.Sequential(*components)

    def forward(self, X: Tensor) -> Any:
        return self.block(X)


class ResidualBlock(FeedForward):

    def forward(self, X: Tensor) -> Any:
        return X + super().forward(X)


class Dataset2Vec(LightningBase):

    def __init__(
        self,
        config: Dataset2VecConfig = Dataset2VecConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
    ):
        super().__init__(optimizer_config)
        self.config = config
        self.output_size = config.output_size
        self._initialize_f(config)
        self._initialize_g(config)
        self._initialize_h(config)

    def _initialize_f(self, config: Dataset2VecConfig) -> None:
        f_components: list[nn.Module] = [
            nn.Linear(2, config.f_dense_hidden_size),
            config.activation_cls(),
        ]
        for _ in range(config.f_block_repetitions):
            f_components.append(ResidualBlock(
                input_size=config.f_dense_hidden_size,
                hidden_size=config.f_res_hidden_size,
                n_layers=config.f_res_n_layers,
                output_size=config.f_dense_hidden_size,
                activation_cls=config.activation_cls,
            ))
        f_components.append(
            nn.Linear(config.f_dense_hidden_size, config.f_out_size)
        )
        self.f = nn.Sequential(*f_components)

    def _initialize_g(self, config: Dataset2VecConfig) -> None:
        g_components: list[nn.Module] = [
            nn.Linear(config.f_out_size, config.g_layers_sizes[0]),
            config.activation_cls(),
        ]
        for prev, curr in zip(
            config.g_layers_sizes[:-1], config.g_layers_sizes[1:]
        ):
            g_components.append(nn.Linear(prev, curr))
            g_components.append(config.activation_cls())
        self.g = nn.Sequential(*g_components)

    def _initialize_h(self, config: Dataset2VecConfig) -> None:
        h_components: list[nn.Module] = [
            nn.Linear(config.g_layers_sizes[-1], config.h_dense_hidden_size),
            config.activation_cls(),
        ]
        for _ in range(config.h_block_repetitions):
            h_components.append(ResidualBlock(
                input_size=config.h_dense_hidden_size,
                hidden_size=config.h_res_hidden_size,
                n_layers=config.h_res_n_layers,
                output_size=config.h_dense_hidden_size,
                activation_cls=config.activation_cls,
            ))
        h_components.append(
            nn.Linear(config.h_dense_hidden_size, config.output_size)
        )
        self.h = nn.Sequential(*h_components)

    def forward(self, X: Tensor, y: Tensor) -> Any:
        assert X.shape[0] == y.shape[0], \
            "X and y must have the same dimensionality"
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        pairs = self._generate_feature_target_pairs(X, y)
        inter_enc = mean(self.f(pairs), dim=1)
        joint_enc = mean(self.g(inter_enc), dim=0)
        return self.h(joint_enc)

    def _generate_feature_target_pairs(
        self, X: Tensor, y: Tensor
    ) -> Tensor:
        X_proc = X.T.repeat_interleave(y.shape[1], dim=0)
        y_proc = y.T.repeat(X.shape[1], 1)
        return stack((X_proc, y_proc), 2)

    def calculate_loss(
        self, labels: Tensor, similarities: Tensor
    ) -> Tensor:
        same = torch.where(labels == 1)[0]
        different = torch.where(labels == 0)[0]
        return -(
            torch.log(similarities[same]).mean()
            + torch.log(1 - similarities[different]).mean()
        )
