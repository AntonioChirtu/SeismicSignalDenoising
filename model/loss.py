import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

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
        # print("Probability: ", log_likelihood.max())
        # print("Target: ", target.max())
        # print("Loss: ", loss)

        return loss
