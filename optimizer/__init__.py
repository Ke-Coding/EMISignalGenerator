from .adamw import AdamW as adamw
from torch.optim import SGD as sgd
from torch.optim import Adam as adam

from utils import param_group_all


def get_optm(op_cfg, net):
    if op_cfg.get('nowd', False):
        nowd_dict = {
            'conv_b': {'weight_decay': 0.0},
            'linear_b': {'weight_decay': 0.0},
            'bn_w': {'weight_decay': 0.0},
            'bn_b': {'weight_decay': 0.0},
        }
        if 'nowd_dict' in op_cfg:
            nowd_dict.update(op_cfg['nowd_dict'])
        op_cfg.kwargs.params, _ = param_group_all(model=net, nowd_dict=nowd_dict)
    else:
        op_cfg.kwargs.params = filter(lambda p: p.requires_grad, net.parameters())
    return globals()[op_cfg.name](**op_cfg.kwargs)
