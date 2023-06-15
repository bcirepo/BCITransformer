
import torch
from torch.utils.data import Dataset
import torch.nn.functional
from pathlib import Path
from sklearn.model_selection import KFold

import argparse
import numpy as np


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        data = self.x[idx]
        data[np.isnan(data)] = 0
        data = torch.tensor(data, dtype=torch.float32)

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return data, label


class LOSOCVDataset:
    def __init__(self, args, eval_subj_index, load_train=True):
        self.args = args
        self.checker(eval_subj_index, self.args.subjects)
        self.args.subjects.remove(eval_subj_index)

        if load_train:
            self._train_data = np.empty((0, self.args.window_len, self.args.eeg_ch))
            self._train_label = np.empty((0,1))

            for subject in self.args.subjects:
                self._train_data = np.vstack((self._train_data, np.load(str(self.args.dataset_dir / ("subj_" + str(subject) + "_data.npy")))[:,:self.args.window_len,:]))
                self._train_label = np.vstack((self._train_label, np.load(str(self.args.dataset_dir / ("subj_" + str(subject) + "_label.npy")))))

        self._eval_data = np.load(str(self.args.dataset_dir / ("subj_" + str(eval_subj_index) + "_data.npy")))[:,:self.args.window_len,:]
        self._eval_label = np.load(str(self.args.dataset_dir / ("subj_" + str(eval_subj_index) + "_label.npy")))

    def get_eval_data_label(self):
        return self._eval_data, self._eval_label

    def checker(self, loso, subject):
        try:
            if loso in subject:
                pass
            else:
                raise Exception(f'{loso} is invalid parameter')

        except ValueError:
            print("List does not contain value")


class TenFCVDataset:
    def __init__(self, subject, args, fold):
        self.args = args
        if subject not in args.subjects: raise ValueError('Subject is INVALID')
        self._fold = fold-1
        self.kfold = KFold(n_splits=10, shuffle=True, random_state=200)
        self._train_data = np.empty((0, self.args.window_len, self.args.eeg_ch))
        self._train_label = np.empty((0, 1))
        self._test_data = np.empty((0, self.args.window_len, self.args.eeg_ch))
        self._test_label = np.empty((0, 1))

        self.get_subject_data(subject)
        self._train_data = np.vstack((self._train_data, self._subject_dataset[self._train_idx]))
        self._train_label = np.vstack((self._train_label, self._subject_label[self._train_idx]))
        self._test_data = np.vstack((self._test_data, self._subject_dataset[self._test_idx]))
        self._test_label = np.vstack((self._test_label, self._subject_label[self._test_idx]))

    def get_data(self):
        return self._train_data, self._train_label, self._test_data, self._test_label

    def get_subject_data(self, subject):
        self._subject_dataset = np.load(str(self.args.dataset_dir / ("subj_" + str(subject) + "_data.npy")))[:,
                                :self.args.window_len, :]
        self._subject_label = np.load(str(self.args.dataset_dir / ("subj_" + str(subject) + "_label.npy")))
        self.kfold.get_n_splits(self._subject_dataset.shape[0])
        self.get_idx()

    def get_idx(self):
        test_idx = []
        data_idx = list(np.linspace(start=0, stop=self._subject_dataset.shape[0] - 1,
                                    num=self._subject_dataset.shape[0], dtype=int))
        for train_index, test_index in self.kfold.split(self._subject_dataset):
            test_idx.append(test_index)

        self._test_idx = test_idx[self._fold]
        self._train_idx = np.array([e for e in data_idx if e not in self._test_idx])

class ArgCenter:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_arg(self):
        return self.lee()


    def lee(self):
        parser = argparse.ArgumentParser(add_help=False)

        parser.add_argument('--d_model', default=186, type=int)
        parser.add_argument('--seq_len', default=65, type=int)
        parser.add_argument('--sliding_window', default=10, type=int)
        parser.add_argument('--patch_len', default=3, type=int)
        parser.add_argument('--dropout', default=0.2, type=float)
        parser.add_argument('--device', default=torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
                            type=str)
        parser.add_argument('--num_layers', default=3, type=int)
        parser.add_argument('--nhead', default=6, type=int)
        parser.add_argument('--category', default=2, type=int)
        parser.add_argument('--norm', default=None, type=bool)
        parser.add_argument('--mode1', default='global', type=str)
        parser.add_argument('--mode2', default='global', type=str)
        parser.add_argument('--train_mode', default='llt', type=str)
        parser.add_argument('--eeg_ch', default=62, type=int)
        parser.add_argument('--query', default=0, type=int)

        parser.add_argument('--window_len', default=999, type=int)
        parser.add_argument('--val_len', default=0, type=int)

        parser.add_argument('--epochs', default=2000, type=int)
        parser.add_argument('--lr', default=0.0005, type=float)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--num_workers', default=1, type=int)

        parser.add_argument('--paradigm', default='gen', type=str)
        parser.add_argument('--eval_subject', default=1, type=int)
        parser.add_argument('--eval_idx', default=0, type=int)
        parser.add_argument('--dataset', default='Lee', type=str)
        parser.add_argument('--dataset_dir', default=Path(__file__).parents[0].resolve() / 'dataset', type=str)

        subjects = list(np.linspace(1, 54, 54, dtype=int))
        parser.add_argument('--subjects', default=subjects, type=list)

        args = parser.parse_args()
        return args


if __name__ == '__main__':
    pass
