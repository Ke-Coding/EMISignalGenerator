import torch.nn as nn


# class MBConv(nn.Module):
#     def __init__(self, cin, ex, cout, s, af, trans: bool):
#         super(MBConv, self).__init__()
#         self.using_skip = (s == 1 and cin == cout)
#         self.trans = trans
#         ConvLayer = nn.ConvTranspose1d if trans else nn.Conv1d
#         ex_ch = round(cin * ex)
#         ker = 5
#         pad = (ker - 1) // 2
#         if trans:
#             ker += s // 2   # todo: stride=4 还会出错，pad都不知道怎么调了
#         self.conv = ConvLayer(cin, cout, ker, s, pad, bias=False)
#         self.bn = nn.BatchNorm1d(cout)
#         self.af = af
#         self.k, self.s, self.p = ker, s, pad
#
#     def forward(self, x):
#         residual = x
#         out = self.af(self.bn(self.conv(x)))
#         if self.using_skip:
#             out += residual
#         print(f'({self.k}, {self.s}, {self.p}): ', x.shape, '=>', out.shape)
#         return out


class MBConv(nn.Module):
    def __init__(self, cin, ex, cout, s, af, trans: bool):
        super(MBConv, self).__init__()
        self.using_skip = (s == 1 and cin == cout)
        ConvLayer = nn.ConvTranspose1d if trans else nn.Conv1d
        ex_ch = round(cin * ex)
        ker = 5
        pad = (ker - 1) // 2
        if trans:
            ker += s // 2   # todo: stride=4 还会出错，pad都不知道怎么调了
        self.pw1 = nn.Sequential(
            ConvLayer(cin, ex_ch, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm1d(ex_ch),
        )
        self.dw = nn.Sequential(
            ConvLayer(ex_ch, ex_ch, kernel_size=ker, padding=pad, stride=s, groups=ex_ch, bias=False),
            nn.BatchNorm1d(ex_ch),
        )
        self.pw2 = nn.Sequential(
            ConvLayer(ex_ch, cout, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm1d(cout)
        )
        self.af = af

    def forward(self, x):
        residual = x
        out = self.af(self.pw1(x))
        out = self.af(self.dw(out))
        out = self.pw2(out)
        if self.using_skip:
            out.add_(residual)
        return out
