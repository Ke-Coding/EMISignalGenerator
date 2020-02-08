import torch
import torch.nn.functional as F


class LabelSmoothCELoss(torch.nn.Module):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes

        self.logsoft = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1-self.smooth_ratio+self.v)

        loss = - torch.sum(self.logsoft(input) * (one_hot.detach())) / input.size(0)
        return loss


def vae_loss(x, x_rec, mu, log_sigma):
    # mce_loss = F.mse_loss(x_rec, x)
    # bce_loss = 15 * F.binary_cross_entropy(x_rec, x, reduction='mean')
    l1 = F.kl_div(x_rec, x, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + log_sigma - mu*mu - log_sigma.exp())
    return l1, kld_loss
