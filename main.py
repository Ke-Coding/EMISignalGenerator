from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import torch
import yaml

from easydict import EasyDict
from tensorboardX import SummaryWriter

from dataset import get_dataloaders
from utils import create_exp_dir, create_logger, set_seed
from solver import VAESolver, ClsSolver


def parse_args_cfg():
    parser = argparse.ArgumentParser(description='EMI Signals Augmentation Experiment')
    parser.add_argument('--cfg_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--vae_load_dir', type=str, default=None)
    parser.add_argument('--cls_load_dir', type=str, default=None)
    parser.add_argument('--num_gpu', type=int, required=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--sig_len', required=True, type=int)
    parser.add_argument('--sig_classes', required=True, type=int)
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
    
    cfg.vae.load_dir, cfg.vae.save_dir = args.vae_load_dir, args.save_dir
    cfg.cls.load_dir, cfg.cls.save_dir = args.cls_load_dir, args.save_dir
    cfg.vae.using_gpu = cfg.cls.using_gpu = torch.cuda.is_available()
    if not cfg.vae.using_gpu:
        print('==> No available GPU device!\n')

    return args, cfg


def get_loggers(args, cfg):
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
    args, cfg = parse_args_cfg()
    lg, tb_lg = get_loggers(args, cfg)
    (vae_train_loader, vae_test_loader, cls_train_loader, cls_test_loader,
     classes) = get_dataloaders(
        data_dir=args.data_dir,
        num_classes=args.sig_classes,
        vae_batch_size=cfg.vae.batch_size,
        cls_batch_size=cfg.cls.batch_size,
        lg=lg
    )

    if args.only_plt:
        pass
    elif args.only_val:
        pass
    else:
        cfg.vae.model.kwargs.input_size = args.sig_len
        cfg.vae.model.kwargs.input_ch = cfg.input_ch
        cfg.vae.model.kwargs.embedding_ch = cfg.embedding_ch
        vae_solver = VAESolver(
            train_loader=vae_train_loader,
            test_loader=vae_test_loader,
            cfg=cfg.vae,
            lg=lg,
            tb_lg=tb_lg
        )
        vae_solver.train_solver()

        cfg.cls.model.kwargs.input_ch = cfg.input_ch
        cfg.cls.model.kwargs.embedding_ch = cfg.embedding_ch
        cfg.cls.model.kwargs.num_classes = args.sig_classes
        cls_solver = ClsSolver(
            train_loader=cls_train_loader,
            test_loader=cls_test_loader,
            encoder=vae_solver.get_encoder(),
            cfg=cfg.cls,
            lg=lg,
            tb_lg=tb_lg
        )
        cls_solver.train_solver()
    
    tb_lg.close()
    lg.info('==> main returned.')


if __name__ == '__main__':
    main()
