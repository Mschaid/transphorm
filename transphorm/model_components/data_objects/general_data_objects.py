from dataclasses import dataclass


@dataclass
class HyperParams:
    batch_size: int = 32
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-8


base_hparams = HyperParams()
