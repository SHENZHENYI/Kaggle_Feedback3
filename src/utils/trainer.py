import os
import time 
import logging
import gc
import torch
from torch import nn
import pandas as pd
from typing import Callable, Dict, List, Tuple, Union

from src.utils.train_utils import AverageMeter, KeepAll, to_cuda, count_parameters

class Trainer:
    """wrap the training process
    """
    def __init__(
        self,
        cfg: object,
        model: nn.Module,
        out_dir: str,
        fold: int,
        train_samples: Union[List, pd.DataFrame] = None,
        val_samples: Union[List, pd.DataFrame] = None,
        test_samples: Union[List, pd.DataFrame] = None,
        device: str = 'cpu'
        ):

        self.steps_done = 0
        self.epochs_done = 0
        self.cur_score = None
        self.best_score = float("inf")
        self.train_loader = None 
        self.val_loader = None 
        self.test_loader = None
        self.scores = []
        self.train_losses = []
        self.val_losses = []

        self.cfg = cfg
        self.fold = fold
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.device = device
        self.model = model.to(device)
        self.optimizer = self.get_optimizer(self.model)
        self.criterion = self.get_criterion(self.model)
        self.metric = self.get_metric(self.model)
        num_training_steps = cfg.epoch * len(train_samples) // cfg.batch_size
        self.scheduler = self.get_scheduler(self.model, self.optimizer, num_training_steps)
        if cfg.apex:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        log_file = os.path.join(out_dir, f"trainer_logs.txt")
        self.logger = self.get_logger(log_file)

        self.logger.info(f'Model has {count_parameters(self.model)} parameters')
        self.logger.info(f'Model {self.model}')


    @staticmethod
    def get_logger(log_file: str):
        logger = logging.getLogger("trainer")
        handler = logging.FileHandler(log_file, mode="w")
        fmt = logging.Formatter("")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger


    ########################
    # DataLoader Methods
    ########################

    @staticmethod
    def get_train_loader(df: pd.DataFrame, model: nn.Module) -> torch.utils.data.DataLoader:
        train_loader = None
        if hasattr(model, "get_train_loader"):
            train_loader = model.get_train_loader(df)
        return train_loader


    @staticmethod
    def get_val_loader(df: pd.DataFrame, model: nn.Module) -> torch.utils.data.DataLoader:
        val_loader = None
        if hasattr(model, "get_val_loader"):
            val_loader = model.get_val_loader(df)
        return val_loader

    def get_test_loader():
        pass 

    ########################
    # Training Methods
    ########################

    @staticmethod
    def _model_train_step(
        model: nn.Module, batch: Dict, criterion: nn.Module,
    ) -> Tuple[Dict, Dict]:
        if hasattr(model, "train_step"):
            return model.train_step(batch, criterion)
        raise NotImplementedError


    def _optimize(
        self,
        batch: Dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: object,
        criterion: nn.Module,
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, List],
    ) -> Tuple[Dict, Dict]:
        with torch.cuda.amp.autocast(enabled=self.cfg.apex):
            outputs_dict, losses_dict = self._model_train_step(model, batch, criterion)

        scaler.scale(losses_dict["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        losses_dict["amp_scaler"] = scaler.get_scale()

        if scheduler is not None:
            scheduler.step()

        losses_dict_detached = self._detach_loss_dict(losses_dict)

        optimizer.zero_grad(set_to_none=True)
        return outputs_dict, losses_dict_detached


    def train_step(self, batch: Dict) -> Tuple[Dict, Dict]:
        outputs, losses = self._optimize(batch, self.model, self.optimizer, self.scaler, \
                                    self.criterion, self.scheduler)
        
        # just in case
        self.model.zero_grad(set_to_none=True)
        self.steps_done += 1
        return outputs, losses


    def train_epoch(self) -> None:
        self.model.train()
        train_losses_epoch = AverageMeter()
        one_percent_step_idx = len(self.train_loader) // 100
        start = time.time()
        for cur_step, batch in enumerate(self.train_loader):
            for k, v in batch.items():
                batch[k] = to_cuda(v, self.cfg.device)
            _, losses = self.train_step(batch)
            train_losses_epoch.update(losses['loss'])
            if cur_step % one_percent_step_idx == 0: # every 1% of total steps
                self.logger.info(f'Batch{cur_step}: train_loss={losses["loss"]}')

        self.train_losses.append(train_losses_epoch.avg)
        epoch_time = time.time() - start
        self.logger.info(f'Epoch{self.epochs_done}: train_loss={train_losses_epoch.avg}; {epoch_time} seconds')
        torch.cuda.empty_cache()
        gc.collect()

    ########################
    # Evaluation Methods
    ########################

    @staticmethod
    def _model_eval_step(
        model: nn.Module, batch: Dict, criterion: nn.Module,
    ) -> Tuple[Dict, Dict]:
        if hasattr(model, "eval_step"):
            return model.eval_step(batch, criterion)
        raise NotImplementedError


    def val_step(self, batch: Dict) -> Tuple[Dict, Dict]:
        with torch.no_grad():
            outputs, losses = self._model_eval_step(self.model, batch, self.criterion)
        losses = self._detach_loss_dict(losses)
        return outputs, losses


    def val_epoch(self) -> None:
        self.model.eval()
        all_predictions = KeepAll()
        all_groudtruths = KeepAll()
        val_losses_epoch = AverageMeter()
        one_percent_step_idx = len(self.val_loader) // 100

        start = time.time()
        for cur_step, batch in enumerate(self.val_loader):
            for k, v in batch.items():
                batch[k] = to_cuda(v, self.cfg.device)
            outputs, losses = self.val_step(batch)
            all_predictions.add_batch(outputs['labels'])
            all_groudtruths.add_batch(batch['labels'])
            val_losses_epoch.update(losses['loss'])
            if cur_step % one_percent_step_idx == 0: # every 1% of total steps
                self.logger.info(f'Batch{cur_step}: val_loss={losses["loss"]}')


        epoch_time = time.time() - start

        score = self.metric(torch.stack(all_groudtruths.all).numpy(),
                            torch.stack(all_predictions.all).numpy())
        self.cur_score = score
        
        self.scores.append(score)
        self.val_losses.append(val_losses_epoch.avg)

        self.logger.info(f'Epoch{self.epochs_done}: loss={val_losses_epoch.avg}; eval score={score}; {epoch_time} seconds')
        torch.cuda.empty_cache()
        gc.collect()


    def fit(self) -> None:
        """train and evaluate
        """
        self.train_loader = self.get_train_loader(self.train_samples, self.model)
        self.val_loader = self.get_val_loader(self.val_samples, self.model)
        for epoch in range(self.cfg.epoch):
            self.train_epoch()
            self.val_epoch()
            self.epochs_done = epoch+1
            if self.cur_score <= self.best_score:
                self.best_score = self.cur_score
                self.save_best_model()
                self.logger.info(f'Epoch{epoch} - Save Best Score: {self.best_score:.4f}')


    def save_best_model(self):
        torch.save({'model': self.model.state_dict(),
                    'score': self.best_score},
                    os.path.join(self.cfg.training_dir, f"{self.cfg.model.replace('/', '-')}_fold{self.fold}_best.pth"))


    @staticmethod
    def get_optimizer(model) -> torch.optim.Optimizer:
        optimizer = None
        if hasattr(model, "get_optimizer"):
            optimizer = model.get_optimizer()
        return optimizer


    @staticmethod
    def get_scheduler(model: nn.Module, optimizer: object, num_train_steps: int) -> object:
        scheduler = None
        if hasattr(model, "get_scheduler"):
            scheduler = model.get_scheduler(optimizer, num_train_steps)
        return scheduler


    def get_lr():
        pass 


    @staticmethod
    def get_criterion(model: nn.Module) -> nn.Module:
        criterion = None
        if hasattr(model, "get_criterion"):
            criterion = model.get_criterion()
        return criterion


    @staticmethod
    def get_metric(model: nn.Module) -> object:
        metric = None
        if hasattr(model, "get_metric"):
            metric = model.get_metric()
        return metric


    ########################
    # Helper Functions
    ########################

    @staticmethod
    def _detach_loss_dict(loss_dict: Dict) -> Dict:
        loss_dict_detached = {}
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                loss_dict_detached[key] = value
            else:
                loss_dict_detached[key] = value.detach().clone()
        return loss_dict_detached