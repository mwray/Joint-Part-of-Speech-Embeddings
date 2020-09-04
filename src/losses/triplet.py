import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Basic triplet loss which calculates the distance between an query and its
    positive and negative examples.
    """
    def __init__(self, margin, weight, reduction='mean'):
        """
        Inputs:
        - margin: value to use as the margin (lower bound between positive and
          negative distances).
        - weight: the weight for this triplet loss.
        - reduction: ['mean', 'sum', 'none'] what reduction to use on the final
          loss calculation.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.weight = weight
        assert reduction in {'mean', 'sum', 'none'}
        if reduction == 'mean':
            self.reduction = lambda x: x.mean()
        elif reduction == 'sum':
            self.reduction = lambda x: x.sum()
        elif reduction == 'none':
            self.reduction = lambda x: x

    def forward(self, x, pos, neg):
        d_pos = (x - pos).pow(2).sum(1)
        d_neg = (x - neg).pow(2).sum(1)
        losses =  F.relu(self.margin + d_pos - d_neg)
        return self.weight * self.reduction(losses)

if __name__ == '__main__':
    triplet_loss_m = TripletLoss(0.1, 1.0)
    triplet_loss_m_0_1 = TripletLoss(0.1, 0.1)
    triplet_loss_s = TripletLoss(0.1, 1.0, 'sum')
    triplet_loss_n = TripletLoss(0.1, 1.0, 'none')

    from datasets import to_tensor
    xs = to_tensor(np.random.rand(64, 256))
    pos = to_tensor(np.random.rand(64, 256))
    neg = to_tensor(np.random.rand(64, 256))

    loss_m = triplet_loss_m(xs, pos, neg)
    loss_m_0_1 = triplet_loss_m_0_1(xs, pos, neg)
    loss_s = triplet_loss_s(xs, pos, neg)
    loss_n = triplet_loss_n(xs, pos, neg)

    assert 0.1 * loss_m == loss_m_0_1

    assert loss_n.mean() == loss_m
    assert loss_n.sum() == loss_s

    assert loss_m.shape == loss_s.shape
    assert loss_n.shape == th.Size([64])
    assert loss_m.shape == th.Size([])
