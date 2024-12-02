import pandas as pd
import numpy as np
import os

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_train(train_split, labeled_ratio=0.2, seed=1):
    df = pd.read_csv(train_split)
    np.random.seed(seed)
    all_ids = df['train'].values.tolist()
    num_labeled = int(np.ceil(labeled_ratio * len(all_ids)))
    train_ids = np.random.choice(all_ids, num_labeled, replace=False)
    unlabeled_ids = np.setdiff1d(all_ids, train_ids)

    train_ids = pd.Series(train_ids)
    paddings = pd.Series([np.nan] * (len(all_ids) - num_labeled))
    train_ids = pd.concat([train_ids, paddings], ignore_index=True)
    df['train'] = train_ids

    unlabeled_ids = pd.Series(unlabeled_ids)
    paddings = pd.Series([np.nan] * num_labeled)
    unlabeled_ids = pd.concat([unlabeled_ids, paddings], ignore_index=True)
    df['unlabeled'] = unlabeled_ids

    df.dropna(subset=['train', 'val', 'test', 'unlabeled'], how='all')

    split_name = os.path.splitext(train_split)
    new_name = f"{split_name[0]}_al{split_name[1]}"
    df.to_csv(new_name, index=False)

    return new_name


def train_loop_al(epoch, model, loader, writer=None):
    with torch.no_grad():

        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        pass
