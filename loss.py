import torch.nn as nn
import torch
import numpy as np

class BCELossWithLength(nn.Module):

    def __init__(self):
        super(BCELossWithLength, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, output, label, length):
        assert output.size() == label.size()
        #BCE
        loss = -(label * torch.log(output) + (1 - label) * torch.log(1 - output))
        mask = torch.zeros(output.size()).to(self.device)

        for idx, l in enumerate(length):
            mask[idx,:l] = 1

        loss = loss * mask

        return torch.sum(loss) / torch.sum(length)
