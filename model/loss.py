import torch.nn.functional as F
import torch


# see https://pytorch.org/docs/master/notes/extending.html


def nll_loss(output, target):
    return F.nll_loss(output, target)


def softCrossEntropy(outputs, labels):
    _, class_num, sample_num = labels.shape
    loss = - torch.sum(torch.mul(labels, torch.log(outputs))) / sample_num
    return loss
