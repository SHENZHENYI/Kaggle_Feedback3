import os
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

@dataclass
class SimpleRegressorConfig():
    seed: int = 42
    n_fold: int = 5
    target_columns: List[str] = field(default_factory=lambda: ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'])
    data_dir: str = './data/fb3'
    save_dir: str = './outputs'
    load_dir: str = './outputs'
    model: str = "microsoft/deberta-v3-base"
    criterion: str = 'l1'
    plm_size: int = 768
    gradient_checkpointing: bool = False
    apex: bool = True
    num_workers: int = 4
    epoch: int = 5
    batch_size: int = 8
    max_len: int = 512
    encoder_lr: float = 2e-5
    decoder_lr: float = 2e-5
    weight_decay: float = 0.01
    eps: float = 1e-6
    betas: Tuple[float] = (0.9, 0.999)
    scheduler: str = 'cosine'
    warmup_ratio: float = 0.0
    num_cycles: float = 1.0
    print_freq: int = 20
    device: str = 'cpu'
    max_grad_norm: float = 1000



