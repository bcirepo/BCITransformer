import torch
from torch.utils.data import Dataset
import torch.nn.functional
from pathlib import Path

import argparse
import numpy as np
from einops import repeat

from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, directory, idx):
        super(MyDataset, self).__init__()
        self.dir = directory
        self.idx = idx

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, i):
        sid = np.load(f'{self.dir}/sid/T{self.idx[i]}_S.npy')

        data = np.load(f'{self.dir}/trial/T{self.idx[i]}_D.npy')
        label = np.load(f'{self.dir}/label/T{self.idx[i]}_L.npy')
        ref = np.load(f'{self.dir}/reference/T{self.idx[i]}_R.npy')

        data[np.isnan(data)] = 0
        data = torch.tensor(data, dtype=torch.float32)

        ref[np.isnan(ref)] = 0
        ref = torch.tensor(ref, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.long)
        sid = torch.tensor(sid, dtype=torch.long)
        return data, label, sid, ref



class ArgCenter:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_arg(self):
        if self._dataset == 'Lee':
            return self.lee()
        else:
            raise 'Dataset Name is INVALID'

    def lee(self):
        parser = argparse.ArgumentParser(add_help=False)

        parser.add_argument('--device', default=torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
                            type=str)
        parser.add_argument('--category', default=2, type=int)
        parser.add_argument('--eeg_ch', default=62, type=int)
        parser.add_argument('--alpha', default=1., type=float)

        parser.add_argument('--window_len', default=1000, type=int)
        parser.add_argument('--val_len', default=0, type=int)

        parser.add_argument('--epochs', default=1000, type=int)
        parser.add_argument('--lr', default=0.0005, type=float)
        parser.add_argument('--batch_size', default=40, type=int)
        parser.add_argument('--num_workers', default=6, type=int)

        parser.add_argument('--eval_idx', default=0, type=int)
        parser.add_argument('--eval_subject', default=1, type=int)
        parser.add_argument('--dataset', default='Lee', type=str)
        parser.add_argument('--dataset_dir', default=Path(__file__).parents[0].resolve() / 'dataset', type=str)

        subjects = list(np.linspace(1, 54, 54, dtype=int))
        parser.add_argument('--subjects', default=subjects, type=list)

        parser.add_argument('--wandb', default=False, type=bool)
        parser.add_argument('--wandb_project', default='MyProject', type=str)
        parser.add_argument('--session_id', default=f'{np.random.randint(0, 10000,1)}', type=str)

        args = parser.parse_args()
        return args


if __name__ == '__main__':
    args = ArgCenter('Lee').get_arg()
    lsv = LOSOCVDataset(args=args, eval_subj_index=1)
    x_train, y_train, train_sid, train_ref = lsv.get_train_data_label()
    x_val, y_val, eval_sid, eval_ref = lsv.get_eval_data_label()
    pass
