from utils import LOSOCVDataset, ArgCenter, TenFCVDataset
from utils import MyDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import numpy as np


class Train:
    def __init__(self, dataset):
        self.args = ArgCenter(dataset).get_arg()

    def SVtrain(self, subject, fold):
        self.args.paradigm = 'SVT'
        self.args.device = 'cpu'
        self.args.eval_idx = fold
        self.args.eval_subject = subject
        self.args.epochs = 200
        self.args.train_id = np.random.randint(low=1000, high=9999, size=1)

        tfcv = TenFCVDataset(subject=subject, args=self.args, fold=self.args.eval_idx)
        x_train, y_train, x_val, y_val = tfcv.get_data()

        self.args.val_len = x_val.shape[0]

        train_loader = MyDataset(x_train, y_train)
        val_loader = MyDataset(x_val, y_val)

        train_iter = DataLoader(dataset=train_loader, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_workers)
        val_iter = DataLoader(dataset=val_loader, batch_size=self.args.batch_size * 4, shuffle=False,
                              num_workers=self.args.num_workers)

        trainer = Trainer(self.args)

        self.args.train_mode = 'llt'
        trainer.train(train_iter, val_iter)

        self.args.train_mode = 'hlt'
        acc = trainer.train(train_iter, val_iter)
        report = f'\n{self.args.paradigm},{self.args.dataset},{subject},{self.args.eval_idx},{self.args.train_id},{acc}'
        write_text = open(f'{self.args.dataset}_report.txt', "a")
        write_text.write(report)

    def SItrain(self, subject):
        lsv = LOSOCVDataset(args=self.args, eval_subj_index=subject)
        x_train, y_train = lsv._train_data, lsv._train_label
        x_val, y_val = lsv.get_eval_data_label()

        self.args.paradigm = 'SIT'
        self.args.device = 'cpu'
        self.args.epochs = 1000
        self.args.val_len = x_val.shape[0]
        self.args.eval_subject = subject
        self.args.device = 'cuda:3'
        train_loader = MyDataset(x_train, y_train)
        val_loader = MyDataset(x_val, y_val)

        train_iter = DataLoader(dataset=train_loader, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_workers)
        val_iter = DataLoader(dataset=val_loader, batch_size=self.args.batch_size * 4, shuffle=False,
                              num_workers=self.args.num_workers)

        trainer = Trainer(self.args)

        self.args.train_mode = 'llt'
        trainer.train(train_iter, val_iter)

        self.args.train_mode = 'hlt'
        acc = trainer.train(train_iter, val_iter)
        report = f'\n{self.args.paradigm},{self.args.dataset},{subject},{self.args.eval_idx},{acc}'
        write_text = open(f'{self.args.dataset}_report.txt', "a")
        write_text.write(report)


if __name__ == '__main__':
    Train(dataset='Lee').SItrain(subject=6)
