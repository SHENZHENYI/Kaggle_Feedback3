import torch
import pandas as pd
from torch.utils.data import Dataset


def prepare_input(text, tokenizer, max_len=512):
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class Feedback3Dataset(Dataset):
    def __init__(self, cfg, tokenizer, df: pd.DataFrame, type: str = 'train'):
        self.texts = df['full_text'].values
        self.max_len = cfg.max_len
        self.tokenizer = tokenizer
        self.type = type
        if type in ['train', 'valid']:
            self.labels = df[cfg.target_columns].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """returns a dict of 4 keys: 
        input_ids,
        token_type_ids,
        attention_mask,
        labels
        """
        inputs = prepare_input(self.texts[item], self.tokenizer, self.max_len)
        if self.type in ['train', 'valid']:
            inputs['labels'] = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs