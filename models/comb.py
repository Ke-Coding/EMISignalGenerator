from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from .mbconv import MBConv
from utils import get_af


class FCBlock(nn.Module):
    def __init__(self, ind, oud, af):
        super(FCBlock, self).__init__()
        self.linear = nn.Linear(ind, oud, bias=False)
        self.bn = nn.BatchNorm1d(oud)
        self.af = af
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.af(x)
        return x


class FCComb(nn.Module):
    def __init__(self, encoder: nn.Module, input_size, num_classes,
                 hid_dims=[128, 72, 48, 32],
                 dropout_p=None,
                 af_name='relu'
                 ):
        r"""
        :param encoder(nn.Module): a complete-trained encoder
        :param input_size(int): input dimension
        :param num_classes(int): output dimension
        :param hid_dims(List[int]): hidden layers' dimension
            - len(hid_dims): the number of the hidden layers
        """
        super(FCComb, self).__init__()
        
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.input_dim, self.output_dim = input_size, num_classes
        self.hid_dims = hid_dims
        self.using_dropout = dropout_p is not None and abs(dropout_p) > 1e-6
        
        self.af = get_af(af_name=af_name)
        
        self.bn0 = nn.BatchNorm1d(input_size)  # this layer has a huge influence on acc
        self.backbone = self._make_backbone(
            len(hid_dims),
            [input_size] + hid_dims[:-1],
            hid_dims
        )
        if self.using_dropout:
            self.dropout_p = dropout_p
            self.dropout = nn.Dropout(p=dropout_p)
        self.svm = nn.Linear(hid_dims[-1], num_classes, bias=True)
    
    def forward(self, x):
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size(0), -1)  # x.size(0): batch size(the number of images in each mini-batch)
        embedding = self.bn0(embedding)
        feature = self.backbone(embedding)
        if self.using_dropout:
            feature = self.dropout(feature)
        logits = self.svm(feature)
        return logits
    
    def _make_backbone(self, num_layers, in_dims, out_dims):
        backbone = OrderedDict()
        name_prefix = 'linear'
        for i in range(num_layers):
            name = f'{name_prefix}_{i}'
            backbone[name] = FCBlock(ind=in_dims[i], oud=out_dims[i], af=self.af)
        return nn.Sequential(backbone)


class CNNComb(nn.Module):
    def __init__(
        self,
        encoder,
        input_ch,
        num_classes,
        channels,
        strides,
        last_ch,
        af_name='relu6',
        dropout_p=None
    ):
        super(CNNComb, self).__init__()
        assert len(channels) == len(strides)

        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.af = get_af(af_name=af_name)
        self.back_bone = self._make_backbone(
            cins=[input_ch] + channels[:-1],
            couts=channels,
            strides=strides,
        )
        self.last_conv = nn.Sequential(
            nn.Conv1d(channels[-1], last_ch, 1, bias=False),
            nn.BatchNorm1d(last_ch)
        )
        
        self.using_dropout = dropout_p is not None and abs(dropout_p) > 1e-6
        if self.using_dropout:
            self.dropout = nn.Dropout(p=dropout_p, inplace=True)
        self.classifier = nn.Linear(last_ch, num_classes)
    
    def _make_backbone(self, cins, couts, strides):
        assert len(cins) == len(couts) == len(strides)
        backbone = OrderedDict()
        name_prefix = 'bottleneck'
        for i in range(len(cins)):
            name = f'{name_prefix}_{i}'
            backbone[name] = MBConv(
                cin=cins[i], ex=3, cout=couts[i], s=strides[i],
                af=self.af, trans=False
            )
        return nn.Sequential(backbone)
    
    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        embedding = self.encoder(x)
        feature = self.back_bone(embedding)
        feature = self.af(self.last_conv(feature), inplace=True).mean(dim=[1, 2])
        if self.using_dropout:
            feature = self.dropout(feature)
        feature = feature.view(feature.shape[0], -1)
        logits = self.classifier(feature)
        return logits


if __name__ == '__main__':  # testing
    print('testing:')
    # net: FCNet = FCNet(input_size=64, num_classes=5, dropout_p=0.2)
    # torchsummary.summary(net, (1, 64))
    # net(tc.rand((2, 1, 64), dtype=tc.float32))
