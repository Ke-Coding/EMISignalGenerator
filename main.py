from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import torch
import yaml

from easydict import EasyDict
from tensorboardX import SummaryWriter

from utils import create_exp_dir, create_logger, set_seed


def parse_args_cfg():
    parser = argparse.ArgumentParser(description='EMI signals classification')
    parser.add_argument('--cfg_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--num_gpu', type=int, required=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--input_size', required=True, type=int)
    parser.add_argument('--num_classes', required=True, type=int)
    parser.add_argument('--only_plt', action='store_true', default=False)
    parser.add_argument('--only_val', action='store_true', default=False)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.log_dir, f'ckpts')
    print(f'==> Args: {args}')
    set_seed(args.seed)
    
    with open(args.cfg_dir) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) if hasattr(yaml, 'FullLoader') else yaml.load(f)
        cfg = EasyDict(cfg)
    print(f'==> Raw config: {cfg}')
    
    cfg.model.kwargs.input_size = args.input_size
    cfg.model.kwargs.num_classes = args.num_classes
    
    cfg.optm.name = cfg.optm.name.strip().lower()
    cfg.optm.sche = cfg.optm.sche.strip().lower()
    cfg.optm.base_lr = cfg.optm.lr if cfg.optm.sche == 'con' else cfg.optm.lr / 4
    cfg.optm.min_lr = cfg.optm.lr / 100.

    cfg.using_gpu = torch.cuda.is_available()
    if not cfg.using_gpu:
        print('==> No available GPU device!\n')

    return args, cfg


def build_loggers(args, cfg):
    print('==> Creating dirs ...')
    create_exp_dir(args.log_dir, scripts_to_save=None)
    create_exp_dir(args.save_dir, scripts_to_save=None)
    print('==> Creating dirs complete.\n')
    
    # Logger
    print('==> Creating logger ...')
    lg = create_logger('global', os.path.join(args.log_dir, 'log.txt'))
    print('==> Creating logger complete.\n')
    lg.info(f'==> Final args: {args}\n')
    lg.info(f'==> Final cfg: {cfg}\n')
    tb_lg = SummaryWriter(os.path.join(args.log_dir, 'events'))
    tb_lg.add_text('exp_time', time.strftime("%Y%m%d-%H%M%S"))
    tb_lg.add_text('exp_dir', f'~/{os.path.relpath(os.getcwd(), os.path.expanduser("~"))}')

    return lg, tb_lg


def main():
    load_dir

if __name__ == '__main__':
    main()
