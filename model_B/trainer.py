
from tqdm import tqdm
from einops import repeat
from einops.layers.torch import Rearrange
import warnings

import torch

import torch.nn as nn
import torch.optim as optim

import wandb

from model import Model

from datetime import date, datetime
import numpy as np


warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.seed = torch.seed()
        self.model = Model()

        self.model.to(self.args.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.c_criterion = nn.CrossEntropyLoss(label_smoothing=0.2).to(self.args.device)
        self.d_criterion = nn.CrossEntropyLoss().to(self.args.device)

        self.best_acc = 0
        self.dates = f'{datetime.now()}'
        if self.args.wandb:
            wandb.init(project=self.args.wandb_project, name=f'S{self.args.eval_subject}_{self.args.wandb_session_name}_{self.dates}')

    def train(self, train_iter, val_iter):

        for epoch in range(self.args.epochs):
            y_loss, y_acc, d_loss, d_acc = 0, 0, 0, 0

            self.model.train()

            for source, target, sid, ref in tqdm(train_iter):
                source = source.squeeze().to(self.args.device)
                target = Rearrange('b c d -> (b c d)', c=1)(target).to(self.args.device)
                ref = ref.squeeze().to(self.args.device)
                d = self.generate_domain(sid.squeeze()).to(self.args.device)

                self.optimizer.zero_grad(set_to_none=True)

                y_hat, d_hat = self.model(source, ref)
                closs = self.c_criterion(y_hat, target)
                dloss = self.d_criterion(d_hat, d)

                loss = closs  + dloss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                y_loss += closs.item()
                d_loss += dloss.item()

                d_acc += self.metric(d, d_hat)

            y_train_losses = y_loss / len(train_iter)
            d_train_losses = d_loss / len(train_iter)
            d_train_acc = d_acc / (self.args.val_len/2)

            y_val_losses, d_val_losses, y_val_acc, d_val_acc = self.evaluate(val_iter)

            if self.args.wandb:
                wandb.log({'y_train_loss': y_train_losses,
                           'd_train_loss': d_train_losses,
                           'd_train_acc': d_train_acc,
                           'y_val_acc': y_val_acc,
                           'y_val_loss': y_val_losses,
                           'd_val_loss': d_val_losses,
                           'd_val_acc': d_val_acc,
                           })

            if self.best_acc < y_val_acc:
                self.best_acc = y_val_acc
                self.save_param(model=self.model, filename='Model')

            print('\n ')
            print(f'Epoch:  {epoch}')
            print(
                f'Y_T_loss:{y_train_losses:.4f} \t Y_V_Loss: {y_val_losses:.4f} \t Y_V_Acc: {y_val_acc:.3f} \t '
                f'Best: {self.best_acc:.3f} \n')

        if self.args.wandb:
            wandb.finish()
        return self.best_acc

    @torch.no_grad()
    def evaluate(self, val_iter):
        y_loss, y_acc, d_loss, d_acc = 0, 0, 0, 0
        self.model.eval()
        for source, target, sid, ref in tqdm(val_iter):
            source = source.squeeze().to(self.args.device)
            target = Rearrange('b c d -> (b c d)', c=1)(target).to(self.args.device)
            ref = ref.squeeze().to(self.args.device)
            d = self.generate_domain(sid.squeeze()).to(self.args.device)

            y_hat, d_hat = self.model(source, ref)
            closs = self.c_criterion(y_hat, target)
            dloss = self.d_criterion(d_hat, d)

            y_loss += closs.item()
            d_loss += dloss.item()

            y_acc += self.metric(target, y_hat)
            d_acc += self.metric(d, d_hat)

        y_val_losses = y_loss / len(val_iter)
        d_val_losses = d_loss / len(val_iter)
        y_val_acc = y_acc / self.args.val_len
        d_val_acc = d_acc / (self.args.val_len/2)
        return y_val_losses, d_val_losses, y_val_acc, d_val_acc

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
        torch.save(param, f'weight/{self.args.dataset}_{filename}_{self.args.eval_subject}_{self.args.wandb_session_name}_{self.args.session_id}.pth')

    def load_param(self, filename):
        param = torch.load(f'weight/{self.args.dataset}_{filename}_{self.args.eval_subject}.pth', map_location=self.args.device)
        return param

    def generate_domain(self, sid):
        sid = np.asarray(sid.detach().cpu())
        batch = sid.shape[0]//2

        sid1 = sid[:batch]
        sid2 = sid[batch:]

        label = np.empty((batch, 1))
        for i in range(batch):
            if sid1[i] != sid2[i]:
                label[i] = 1
            else:
                label[i] = 0

        label = torch.tensor(label.squeeze(), dtype=torch.long)
        return label

if __name__ == '__main__':
    pass