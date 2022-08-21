import torch
import torch.nn as nn


class DiceLossAndMSELoss(nn.Module):
    def __init__(self):
        super(DiceLossAndMSELoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        input_flat = torch.sigmoid(input_flat)
        target_flat = target.view(N, -1)
        zeros = torch.zeros(target_flat.shape).to(device='cuda')
        DiffInTar = input_flat - target_flat
        mseloss = nn.MSELoss()
        loss1 = mseloss(input_flat, target_flat)
        TP = torch.max(target_flat - torch.abs(DiffInTar), zeros)
        FP = torch.max(DiffInTar, zeros)
        FN = torch.max(-DiffInTar, zeros)
        # Paper uses -dice_coe, I use 1 - dice_coe
        loss2 = 1 - (2 * TP.sum(1) / (2 * TP.sum(1) + FP.sum(1) + FN.sum(1)))
        loss = loss2.sum() / (N * input_flat.shape[1]) + loss1

        return loss


class BCELossAndDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELossAndDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, **kwargs):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, **kwargs):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # print('logits: {}, targets: {}'.format(logits.size(), targets.size()))
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        # smooth = 1.

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / (num * m1.shape[1])
        return score


class DiceScoreEval(nn.Module):
    def __init__(self):
        super(DiceScoreEval, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        input_flat = torch.sigmoid(input_flat)
        target_flat = target.view(N, -1)
        zeros = torch.zeros(target_flat.shape).to(device='cuda')
        DiffInTar = input_flat - target_flat
        TP = torch.max(target_flat - torch.abs(DiffInTar), zeros)
        FP = torch.max(DiffInTar, zeros)
        FN = torch.max(-DiffInTar, zeros)
        # Paper uses -dice_coe, I use 1 - dice_coe
        loss2 = 1 - (2 * TP.sum(1) / (2 * TP.sum(1) + FP.sum(1) + FN.sum(1)))
        # Loss for per image
        loss = loss2.sum() / N

        return loss
