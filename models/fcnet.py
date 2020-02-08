from collections import OrderedDict
import torch as tc
import torch.nn as nn
import torchsummary

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


class FCNet(nn.Module):
    def __init__(self, input_size, num_classes,
                 hid_dims=[128, 72, 48, 32],
                 dropout_p=None,
                 af_name='relu'
                 ):
        r"""
        :param input_size(int): input dimension
        :param num_classes(int): output dimension
        :param hid_dims(List[int]): hidden layers' dimension
            - len(hid_dims): the number of the hidden layers
        """
        super(FCNet, self).__init__()
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
        self.classifier = nn.Linear(hid_dims[-1], num_classes, bias=True)
    
    def forward(self, x):
        """
        :param x: (3D Tensor, C*W*H), i.e. input(images)
        input(images) ==[view]==> flatten inputs ==[batchnorm0]==> normalized flatten inputs
        ==[backbone]==> features ==[dropout]==> sparse features ==[classifier]==> output(logits)
        """
        flatten_x = x.view(x.size(0), -1)  # x.size(0): batch size(the number of images in each mini-batch)
        flatten_x = self.bn0(flatten_x)
        features = self.backbone(flatten_x)
        if self.using_dropout:
            features = self.dropout(features)
        logits = self.classifier(features)
        return logits
    
    def _make_backbone(self, num_layers, in_dims, out_dims):
        backbone = OrderedDict()
        name_prefix = 'linear'
        for i in range(num_layers):
            name = name_prefix + "_%d" % i
            backbone[name] = FCBlock(ind=in_dims[i], oud=out_dims[i], af=self.af)
        return nn.Sequential(backbone)


if __name__ == '__main__':  # testing
    net: FCNet = FCNet(input_size=64, num_classes=5, dropout_p=0.2)
    torchsummary.summary(net, (1, 64))
    net(tc.rand((2, 1, 64), dtype=tc.float32))
