import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer, \
                         get_cosine_schedule_with_warmup, \
                         get_cosine_with_hard_restarts_schedule_with_warmup, \
                         get_linear_schedule_with_warmup
from typing import Dict, List, Tuple

from models.model import BaseModel
from models.layers import MeanPooling
from data.dataset import collate, Feedback3Dataset
from utils.train_utils import get_scheduler, mcrmse

class BaselineRegressor(BaseModel):
    """A simple regression model using the representations from a pretrained model like DEBERTA
    """
    def __init__(
        self, 
        cfg,
        cfg_path: str = None,
        use_pretrained: bool = None
    ):
        super().__init__()
        if cfg_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.model_config = torch.load(cfg_path)

        if use_pretrained:
            self.transformer = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.transformer = AutoModel(self.model_config)

        if cfg.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()

        self.cfg = cfg
        self.pool = MeanPooling()
        self.fc = nn.Linear(cfg.plm_size, len(cfg.target_columns))
    
    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden_state = self.transformer(input_ids=inputs, attention_mask=attention_mask)[0]
        features = self.pool(last_hidden_state, attention_mask)
        out = self.fc(features)
        return out
    
    def train_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        batch = collate(batch)
        out = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = criterion(out, batch['labels'])
        return {'labels': out}, {'loss': loss}

    def eval_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, torch.Tensor]:
        return self.train_step(batch, criterion)

    def get_train_loader(self, df: pd.DataFrame):
        train_dataset = Feedback3Dataset(self.cfg, self.get_tokenizer(), df, 'train')
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.cfg.batch_size,
                                  shuffle=True,
                                  num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True)
        return train_loader

    def get_val_loader(self, df: pd.DataFrame):
        valid_dataset = Feedback3Dataset(self.cfg, self.get_tokenizer(), df, 'valid')
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.cfg.batch_size*2,
                                  shuffle=False,
                                  num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
        return valid_loader

    def load_checkpoint(self):
        pass


    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model, use_fast=True)
        return tokenizer

    def get_metric(self) -> object:
        """ get a metric function
        mcrmse(target: np.array, pred: np.array) -> float
        """
        return mcrmse

    def get_criterion(self) -> nn.Module:
        if self.cfg.criterion == 'l1':
            return nn.SmoothL1Loss(reduction='mean')
        elif self.cfg.criterion == 'l2':
            return nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError
    
    def get_optimizer(self):
        head_params = list(self.fc.named_parameters())
        param_optimizer = list(self.transformer.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
                "lr": self.cfg.encoder_lr
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.cfg.encoder_lr
            },
            {
                "params": [p for n,p in head_params], 
                "weight_decay": self.cfg.weight_decay,
                "lr": self.cfg.decoder_lr
            },
        ]
        return AdamW(optimizer_parameters, lr=self.cfg.encoder_lr, eps=self.cfg.eps, betas=self.cfg.betas)


    def get_scheduler(self, optimizer: object, num_train_steps: int):
        return  get_scheduler(self.cfg.scheduler, optimizer, num_train_steps, self.cfg.warmup_ratio, self.cfg.num_cycles)
