
from tqdm import tqdm
from einops import repeat
from einops.layers.torch import Rearrange
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from model import Model


warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.seed = torch.seed()
        self.model = Model(self.args).to(self.args.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15).to(self.args.device)

        self.best_acc = 0

    def train(self, train_iter, val_iter):
        if self.args.train_mode == 'llt':
            self.llt_train(train_iter, val_iter)

        if self.args.train_mode == 'hlt':
            self.hlt_train(train_iter, val_iter)
        return self.best_acc

    def llt_train(self, train_iter, val_iter):
        for epoch in range(self.args.epochs):
            batch_loss, batch_acc = 0, 0

            self.model.train()
            for source, target in tqdm(train_iter):
                source = source.to(self.args.device)
                target1 = repeat(target, 'h c -> (h c r)', r=self.args.seq_len).to(self.args.device)

                self.optimizer.zero_grad()
                y1, _, _ = self.model(source)

                loss = self.criterion(y1, target1)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                batch_loss += loss.item()
                batch_acc += self.metric(target1, y1)

            train_loss = batch_loss / len(train_iter)

            val_loss1, val_loss2, val_acc1, val_acc2 = self.evaluate(val_iter)
            val_acc1 = val_acc1 / (self.args.val_len * self.args.seq_len)

            if self.best_acc < val_acc1:
                self.best_acc = val_acc1
                self.save_param(model=self.model, filename='LLT')

            print(f'Epoch:  {epoch}')
            print(
                f'Train Loss Lv 1: {train_loss:.3f} \t Val. Loss Lv 1:'
                f' {val_loss1:.3f} \t Val. Acc Lv 1: {val_acc1: .3f}\n')

    def hlt_train(self, train_iter, val_iter):

        param = self.load_param(filename='LLT')
        self.model.load_state_dict(param['model'])
        self.model.llt_freeze()

        self.best_acc = param['best_acc']

        for epoch in range(self.args.epochs):
            batch_loss, batch_acc = 0, 0

            self.model.train()

            for source, target in tqdm(train_iter):
                source = source.to(self.args.device)
                target = Rearrange('b c -> (b c)', c=1)(target).to(self.args.device)

                self.optimizer.zero_grad()
                _, y, _ = self.model(source)

                loss = self.criterion(y, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                batch_loss += loss.item()
                batch_acc += self.metric(target, y)

            train_loss = batch_loss / len(train_iter)

            val_loss1, val_loss2, val_acc1, val_acc2 = self.evaluate(val_iter)
            val_acc2 = val_acc2 / self.args.val_len

            if self.best_acc < val_acc2:
                self.best_acc = val_acc2
                self.save_param(model=self.model, filename='HLT')

            print('\n ')
            print(f'Epoch:  {epoch}')
            print(
                f'Train Loss Lv 2: {train_loss:.3f} \t Val. Loss Lv 2:'
                f' {val_loss2:.3f} \t Val. Acc Lv 2: {val_acc2: .3f}\n')

        print(f'\nBest Eval. for Subject {self.args.eval_subject} is {self.best_acc}')

    @torch.no_grad()
    def evaluate(self, val_iter):
        batch_loss1, valid_acc1, batch_loss2, valid_acc2 = 0, 0, 0, 0
        self.model.eval()
        for source, target in tqdm(val_iter):
            source = source.to(self.args.device)
            target1 = repeat(target, 'h c -> (h c r)', r=self.args.seq_len).to(self.args.device)
            target2 = Rearrange('b c -> (b c)', c=1)(target).to(self.args.device)

            y1, y2, _ = self.model(source)

            loss1 = self.criterion(y1, target1)
            loss2 = self.criterion(y2, target2)

            batch_loss1 += loss1.item()
            valid_acc1 += self.metric(target1, y1)

            batch_loss2 += loss2.item()
            valid_acc2 += self.metric(target2, y2)

        valid_loss1 = batch_loss1 / len(val_iter)
        valid_loss2 = batch_loss2 / len(val_iter)
        return valid_loss1, valid_loss2, valid_acc1, valid_acc2

    def metric(self, target, output):
        acc = 0
        out = torch.argmax(output, dim=1)
        for i in range(target.shape[0]):
            if target[i] == out[i]:
                acc += 1
        return acc

    def save_param(self, model, filename):
        param = {'model': model.state_dict(),
                 'optim': self.optimizer.state_dict(),
                 'best_acc': self.best_acc,
                 'seed': self.seed}
        torch.save(param, f'weight/{self.args.dataset}_{filename}_{self.args.eval_subject}_{self.args.eval_idx}_{self.args.train_id}.pth')

    def load_param(self, filename):
        param = torch.load(f'weight/{self.args.dataset}_{filename}_{self.args.eval_subject}_{self.args.eval_idx}_{self.args.train_id}.pth')
        return param

    @torch.no_grad()
    def inference(self, source, target):
        param = self.load_param('HLT')
        self.model.load_state_dict(param['model'])
        self.model.eval()

        source = torch.tensor(source, dtype=torch.float32).to(self.args.device)
        target2 = Rearrange('b c -> (b c)', c=1)(torch.tensor(target, dtype=torch.long)).to(self.args.device)
        _, y2, _ = self.model(source)
        valid_acc = self.metric(target2, y2)/source.shape[0]
        return valid_acc

    @torch.no_grad()
    def attention(self, source):
        source = torch.tensor(source, dtype=torch.float32).to(self.args.device)
        param = torch.load('weight/Lee_HLT_2_0.pth')
        self.model.load_state_dict(param['model'])
        self.model.eval()
        _, _, score = self.model(source)
        return score


if __name__ == '__main__':
    pass