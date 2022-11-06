import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from configs.regressor0_cfg import SimpleRegressorConfig
from models.regressors import BaselineRegressor
from utils.trainer import Trainer

def main():
    cfg = SimpleRegressorConfig()
    train_df = pd.read_csv(os.path.join(cfg.data_dir, 'train.csv'))
    Fold = MultilabelStratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train_df, train_df[cfg.target_columns])):
        train_df.loc[val_index, 'fold'] = int(n)
    train_df['fold'] = train_df['fold'].astype(int)

    for fold_id in range(cfg.n_fold):
        fold_train = train_df[train_df['fold'] != fold_id].reset_index(drop=True)
        fold_valid = train_df[train_df['fold'] == fold_id].reset_index(drop=True)

        model = BaselineRegressor(cfg, cfg_path=None, use_pretrained=True)
        
        trainer = Trainer(
            cfg=cfg,
            model=model,
            fold=fold_id,
            out_dir=cfg.training_dir,
            train_samples=fold_train,
            val_samples=fold_valid,
            test_samples=None,
            device=cfg.device
        )

        trainer.fit()
        break



if __name__ == '__main__':
    main()
    print('Done!')