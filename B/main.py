
from utils import ArgCenter
from utils import MyDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import numpy as np


class Train:
    def __init__(self, dataset):
        self.args = ArgCenter(dataset).get_arg()

    def SItrain(self, subject):

        self.args.paradigm = 'SIT'
        self.args.epochs = 100
        self.args.val_len = 400
        self.args.eval_subject = subject
        self.args.device = 'cuda:2'

        self.args.wandb = True
        self.args.wandb_project = 'Domain_Adaptation_ResNet_8A'
        self.args.wandb_session_name = 'Report'

        idx = np.asarray([np.linspace(start=0, stop=400*54-1, num= 400*54, dtype=int)]).squeeze()
        test_idx = np.asarray([np.linspace(start=(subject-1)*400, stop=subject*400-1, num= 400, dtype=int)]).squeeze()
        train_idx = np.delete(idx, test_idx)

        train_loader = MyDataset(self.args.dataset_dir, train_idx)
        val_loader = MyDataset(self.args.dataset_dir, test_idx)

        train_iter = DataLoader(dataset=train_loader, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_workers)
        val_iter = DataLoader(dataset=val_loader, batch_size=self.args.batch_size * 4, shuffle=False,
                              num_workers=self.args.num_workers)

        trainer = Trainer(self.args)
        acc = trainer.train(train_iter, val_iter)
        report = f'\n{self.args.paradigm},{self.args.dataset},{self.args.session_id},{subject},{self.args.wandb_session_name},{acc}'
        write_text = open(f'{self.args.dataset}_report.txt', "a")
        write_text.write(report)


if __name__ == '__main__':
    subject = 1
    print(f'================ Start Training for Subject-{subject} ================')
    Train(dataset='Lee').SItrain(subject=subject)

