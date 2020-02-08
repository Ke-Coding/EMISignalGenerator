from collections import OrderedDict
import torch
import torch.nn as nn
import torchsummary

from .mbconv import MBConv
from utils import get_af


class CNNEncoder(nn.Module):
    def __init__(self, encoder, cin, cout):
        super(CNNEncoder, self).__init__()
        self.encoder = encoder
        self.vae_mu = nn.Conv1d(cin, cout, 1)
        self.vae_log_sigma = nn.Conv1d(cin, cout, 1)
    
    def forward(self, x):
        x = self.encoder(x)
        mu, log_sigma = self.vae_mu(x), self.vae_log_sigma(x)
        sigma = log_sigma.mul(0.5).exp_()
        gaussian = mu + torch.randn_like(sigma) * sigma
        # print('gau: ', gaussian.shape)
        return gaussian, mu, log_sigma


class CNNVAE(nn.Module):
    def __init__(
        self,
        input_ch,
        input_size,
        embedding_ch,
        channels,
        strides,
        af_name='relu6'
    ):
        super(CNNVAE, self).__init__()
        
        assert len(channels) == len(strides)
        self.af = get_af(af_name=af_name)
        self.input_ch = input_ch
        
        self.encoder = CNNEncoder(
            encoder=self._make_backbone(
                cins=[input_ch] + channels[:-1],
                couts=channels,
                strides=strides,
                trans=False
            ),
            cin=channels[-1],
            cout=embedding_ch
        )
        self.decoder = self._make_backbone(
            cins=[embedding_ch] + channels[::-1],
            couts=channels[::-1] + [input_ch],
            strides=[1] + strides[::-1],
            trans=True
        )
    
    def _make_backbone(self, cins, couts, strides, trans: bool):
        assert len(cins) == len(couts) == len(strides)
        backbone = OrderedDict()
        name_prefix = f'bottleneck{"_trans" if trans else ""}'
        for i in range(len(cins)):
            name = f'{name_prefix}_{i}'
            backbone[name] = MBConv(
                cin=cins[i], ex=3, cout=couts[i], s=strides[i],
                af=self.af, trans=trans
            )
        return nn.Sequential(backbone)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_ch, -1)
        gaussian, mu, log_sigma = self.encoder(x)
        x_rec = torch.sigmoid(self.decoder(gaussian))
        x_rec = x_rec.view(batch_size, -1)
        return x_rec, mu, log_sigma
    
    def get_encoder(self):
        return self.encoder


class RNNVAE(nn.Module):
    def __init__(self, ind, oud, af):
        super(RNNVAE, self).__init__()
        self.linear = nn.Linear(ind, oud, bias=False)
        self.bn = nn.BatchNorm1d(oud)
        self.af = af
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.af(x)
        return x
    
    def get_encoder(self):
        return None


if __name__ == '__main__':
    print('test:')
