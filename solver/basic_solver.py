import os
import time
import datetime
import torch
from torch.optim.optimizer import Optimizer

from models import get_model
from optimizer import get_optm
from scheduler import get_sche
from utils import refine_str, AverageMeter


class BasicSolver(object):
    def __init__(self, train_loader, test_loader, cfg, lg, tb_lg):
        
        self.loaded_ckpt = None
        if cfg.load_dir is not None and len(self.cfg.load_dir) > 0:
            ckpt_path = os.path.abspath(self.cfg.load_dir)
            lg.info(f'==> Getting ckpt at {ckpt_path} ...')
            assert os.path.isfile(ckpt_path), '==> Error: no checkpoint file found!'
            self.loaded_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            lg.info(f'==> Getting ckpt complete, last_iter: {self.loaded_ckpt["last_iter"]}, best_acc: {loaded_ckpt["best_acc"]}.\n')
            cfg.sche.kwargs.last_iter = self.loaded_ckpt['last_iter']
        else:
            cfg.sche.kwargs.last_iter = -1

        cfg.sche.name = refine_str(cfg.sche.name)
        cfg.sche.kwargs.T_max = len(train_loader) * cfg.epochs
        cfg.sche.kwargs.base_lr = cfg.optm.kwargs.lr if cfg.sche.name == 'con' else cfg.optm.kwargs.lr / 4
        cfg.sche.kwargs.eta_min = cfg.optm.kwargs.lr / 100.
        cfg.sche.kwargs.warmup_lr = cfg.optm.kwargs.lr
        cfg.sche.kwargs.warmup_steps = max(round(cfg.sche.kwargs.T_max * 0.04), 1)

        cfg.optm.name = refine_str(cfg.optm.name)
        cfg.optm.kwargs.lr = cfg.sche.kwargs.base_lr
        
        self.cfg = cfg
        self.train_loader, self.test_loader = train_loader, test_loader
        self.lg, self.tb_lg = lg, tb_lg
        self.net = self.optm = self.sche = None

    def build_solver(self):
        self.lg.info(f'==> get_model: {self.cfg.model}')
        self.net: torch.nn.Module = get_model(model_cfg=self.cfg.model, loaded_ckpt=self.loaded_ckpt, using_gpu=self.cfg.using_gpu)
        self.lg.info(f'==> get_optm: {self.cfg.optm}')
        self.optm: Optimizer = get_optm(op_cfg=self.cfg.optm, net=self.net)
        self.lg.info(f'==> get_sche: {self.cfg.sche}')
        self.sche = get_sche(sc_cfg=self.cfg.sche, optm=self.optm)
    
    def test_solver(self):
        raise NotImplementedError

    def train_solver(self):
        raise NotImplementedError
