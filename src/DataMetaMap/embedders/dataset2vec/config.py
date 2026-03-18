from typing import Annotated, Type

from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator
from torch import nn
from torch.optim import Adam, Optimizer

from .utils import Validators


class Dataset2VecConfig(BaseModel):
    """Configuration of the Dataset2Vec encoder"""

    activation_cls: Type[nn.Module] = Field(default=nn.ReLU)
    f_dense_hidden_size: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 32
    f_res_hidden_size: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 32
    f_res_n_layers: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 3
    f_block_repetitions: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 7
    f_out_size: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 32
    g_layers_sizes: Annotated[
        list[int],
        AfterValidator(Validators.all_elements_positive),
        AfterValidator(Validators.non_empty),
    ] = [32, 16, 8]
    h_dense_hidden_size: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 16
    h_res_hidden_size: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 16
    h_res_n_layers: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 3
    h_block_repetitions: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 3
    output_size: Annotated[
        int, AfterValidator(Validators.is_positive)
    ] = 16


class OptimizerConfig(BaseModel):
    """Configuration of the Dataset2Vec training"""

    gamma: Annotated[
        float, AfterValidator(Validators.is_positive)
    ] = 1
    optimizer_cls: Type[Optimizer] = Adam
    learning_rate: Annotated[
        float, AfterValidator(Validators.is_positive)
    ] = 1e-4
    weight_decay: Annotated[
        float, AfterValidator(Validators.non_negative)
    ] = 1e-4
