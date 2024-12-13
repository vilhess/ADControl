import torch
import torch.nn as nn

class LossVAE(nn.Module):
    def __init__(self):
        super(LossVAE, self).__init__()

        self.rec_loss = nn.BCELoss(reduction="sum")

    def kl_div(self, mu, logvar):
        loss = torch.mean(-0.5*torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        return loss
    
    def forward(self, x, x_output, mu, logvar):
        rec_loss = self.rec_loss(x_output, x)
        kl = self.kl_div(mu, logvar)
        return rec_loss+kl