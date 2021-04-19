import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def softXEnt(input, target):
    # logprobs = torch.nn.functional.log_softmax(input, dim=2)
    # print(logprobs.max())
    return -(target * input).sum(dim=1) / input.shape[1]


class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        _, class_num, sample_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num

        return loss
