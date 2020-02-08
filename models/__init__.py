from .vae import CNNVAE, RNNVAE
from .comb import FCComb
from .comb import CNNComb

from utils import init_params


def get_model(model_cfg, loaded_ckpt, using_gpu):
    print(f'bulid model: {model_cfg.name}')
    net = globals()[model_cfg.name](**model_cfg.kwargs)
    init_params(net)
    num_para = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'param: {num_para / 1e6} * 10^6')
    if loaded_ckpt is not None:
        net.load_state_dict(loaded_ckpt['model'])
    return net.cuda() if using_gpu else net
