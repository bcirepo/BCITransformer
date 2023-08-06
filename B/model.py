
import torch


import torch.nn as nn
from torch.autograd import Function


class RevGradOptim(Function):

    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_ # a = 0.5
        return grad_input, None


class RevGrad(nn.Module):
    def __init__(self, alpha=.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGradOptim.apply(input_, self._alpha)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Spatial Temporal
        self.eeg_ch = 62
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(20, 1), stride=(1, 1), bias=False),
            # nn.BatchNorm2d(num_features=25, momentum=0.1, affine=True, eps=1e-5),
            # nn.ELU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(1, self.eeg_ch), stride=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=12, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )

        self.blockref = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(num_features=12, momentum=0.1, affine=True, eps=1e-5),
        )

        # temporal axis
        self.block2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(20, 1), stride=(1, 1), bias=False),
            nn.ELU(),
            nn.BatchNorm2d(num_features=24, momentum=0.1, affine=True, eps=1e-5),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
        )

        # temporal axis
        self.block3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(20, 1), stride=(1, 1), bias=False),
            nn.ELU(),
            nn.BatchNorm2d(num_features=48, momentum=0.1, affine=True, eps=1e-5),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
        )

        # temporal axis
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(20, 1), stride=(1, 1), bias=False),
            nn.ELU(),
            nn.BatchNorm2d(num_features=96, momentum=0.1, affine=True, eps=1e-5),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Conv2d(in_channels=96, out_channels=2, kernel_size=(2, 1), bias=True)
        )

        self.revgrad = RevGrad(0.03)
        self.domain_discrimination = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(20, 1), stride=(1, 1), bias=False),
            nn.ELU(),
            nn.BatchNorm2d(num_features=96, momentum=0.1, affine=True, eps=1e-5),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Conv2d(in_channels=96, out_channels=2, kernel_size=(2, 1), bias=True)
        )

    def forward(self, x, ref):
        x = x.unsqueeze(-1)
        x = x.permute(0, 3, 2, 1)

        ref1 = ref[:, 0]
        ref2 = ref[:, 1]
        ref1 = ref1.unsqueeze(-1)
        ref1 = ref1.permute(0, 3, 2, 1)

        ref2 = ref2.unsqueeze(-1)
        ref2 = ref2.permute(0, 3, 2, 1)

        ref1 = self.block1(ref1)
        ref2 = self.block1(ref2)
        x = self.block1(x)

        ref = ref1 + ref2

        xs = ref + x
        xs = self.blockref(xs)

        x = x + xs

        x = self.block2(x)
        x = self.block3(x)

        y_hat = self.classification_head(x)
        y_hat = y_hat.squeeze()

        y_disc = self.revgrad(x)
        y_disc = self.domaindisc(y_disc)
        y_disc = y_disc.squeeze()

        return y_hat, y_disc

    def domaindisc(self, x):
        batch = x.shape[0]//2

        x1 = x[:batch]
        x2 = x[batch:]

        new_x = torch.cat((x1, x2), dim=1)
        s = self.domain_discrimination(new_x)
        return s


if __name__ == '__main__':
    model = Model()
    dummy = torch.zeros((40,62,1000))
    ref = torch.zeros((40,2,62,1000))

    implement = model(dummy, ref)

    pass