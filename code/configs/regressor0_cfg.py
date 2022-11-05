import os
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

@dataclass
class SimpleRegressorConfig():
    seed: int = 1997
    n_fold: int = 5
    target_columns: List[str] = field(default_factory=lambda: ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'])
    data_dir: str = '../data/fb3/'
    training_dir: str = '/Users/zhenyishen/Documents/doc-mba/GitHub/Kaggle_Feedback3/code/outputs'
    model: str = "microsoft/deberta-v3-base"
    criterion: str = 'l1'
    plm_size: int = 768
    gradient_checkpointing: bool = True
    apex: bool = True
    num_workers: int = 4
    epoch: int = 5
    batch_size: int = 2
    max_len: int = 512
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 0.01
    eps: float = 1e-6
    betas: Tuple[float] = (0.9, 0.999)
    scheduler: str = 'cosine'
    warmup_ratio: float = 0.1
    num_cycles: int = 5
    device: str = 'cuda'



