from collections import namedtuple
from typing import Union

ModelParams = namedtuple(
    "ModelParams",
    [
        "name_str",
        "num_features",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "version",
    ],
)


def encode(
    name_str: str,
    num_features: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    version: Union[str, int],
) -> str:
    return ".".join(
        map(
            str,
            [
                name_str,
                num_features,
                d_model,
                nhead,
                num_layers,
                dim_feedforward,
                version,
                "model",
            ],
        )
    )


def decode(name: str) -> ModelParams:
    name_str, num_features, d_model, nhead, num_layers, dim_feedforward, version, _ = (
        name.split(".")
    )
    return ModelParams(
        name_str,
        int(num_features),
        int(d_model),
        int(nhead),
        int(num_layers),
        int(dim_feedforward),
        version,
    )
