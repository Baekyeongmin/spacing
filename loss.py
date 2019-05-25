import torch.nn as nn
import torch


class BCELossWithLength(nn.Module):

    def __init__(self):
        super(BCELossWithLength, self).__init__()

    def forward(self, output, label, length):
        assert output.size() == label.size()
        #BCE
        loss = -(label * torch.log(output) + (1 - label) * torch.log(1 - output))
        mask = torch.zeros(output.size())

        for idx, l in enumerate(length):
            mask[idx,:l] = 1

        loss = loss * mask
        return torch.sum(loss) / torch.sum(length)
