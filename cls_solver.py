from __future__ import print_function
from __future__ import division

import argparse
import datetime
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import yaml

from models import *
from optimizer import AdamW
from loss import LabelSmoothCELoss
from emi_dataset import EMIDataset
from scheduler import CosineLRScheduler, ConstScheduler
from utils import create_exp_dir, create_logger, set_seed, param_group_all, init_params, AverageMeter


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


rank = 0

# Parsing args
args = parser.parse_args()
args.save_dir = os.path.join(args.log_dir, f'ckpts')
if rank == 0:
    print(f'==> Args: {args}')

cfg = None
with open(args.config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader) if hasattr(yaml, 'FullLoader') else yaml.load(f)

if rank == 0:
    print(f'==> Raw config: {cfg}')

cfg.model.kwargs.input_size = args.input_size
cfg.model.kwargs.num_classes = args.num_classes

cfg.optm.name = cfg.optm.name.strip().lower()
cfg.optm.sche = cfg.optm.sche.strip().lower()
cfg.optm.base_lr = cfg.optm.lr if cfg.optm.sche == 'con' else cfg.optm.lr / 4
cfg.optm.min_lr = cfg.optm.lr / 100.

if rank == 0:
    print(f'==> Final config: {cfg}')


# rank, world_size = dist_init()
# local_rank, __, self_group_name, leader_group_name = group_split(
#     global_rank=rank, world_size=world_size,
#     num_groups=args.num_groups, group_size=args.group_size)

rank = 0

if rank == 0:
    print('==> Creating dirs ...')
    create_exp_dir(args.log_dir, scripts_to_save=None)
    create_exp_dir(args.save_dir, scripts_to_save=None)
    print('==> Creating dirs complete.\n')

# Logger
if rank == 0:
    print('==> Creating logger ...')
    lg = create_logger('global', os.path.join(args.log_dir, 'log.txt'))
    print('==> Creating logger complete.\n')
    lg.info(f'==> Final args: {args}\n')
    tb_lg = SummaryWriter(os.path.join(args.log_dir, 'events'))
    tb_lg.add_text('exp_time', time.strftime("%Y%m%d-%H%M%S"))
    tb_lg.add_text('exp_dir', f'~/{os.path.relpath(os.getcwd(), os.path.expanduser("~"))}')
else:
    lg = None
    tb_lg = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
using_gpu = torch.cuda.is_available()
if not using_gpu and rank == 0:
    lg.info('==> No available GPU device!\n')

# Seed
set_seed(args.seed)

# Datasets
if rank == 0:
    lg.info('==> Preparing data..')

if args.data_dir is None or len(args.data_dir) == 0:
    raise AttributeError(f'data file {args.data_dir} not found!')

if rank == 0:
    lg.info(f'==> Reading dataset from {args.data_dir} ...')

train_set = EMIDataset(
    data_dir=args.data_dir, train=True, num_classes=cfg.num_classes, normalize=True)
test_set = EMIDataset(
    data_dir=args.data_dir, train=False, num_classes=cfg.num_classes, normalize=True)

if rank == 0:
    lg.info(f'==> Getting dataloader from {args.data_dir} ...')
train_loader = DataLoader(
    dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(
    dataset=test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

classes = ('pink', 'brown', 'laplace', 'uniform', 'exponential')
assert cfg.num_classes == len(classes)
if rank == 0:
    lg.info(f'==> Preparing data complete, classes:{classes} .\n')

# Load checkpoints.
loaded_ckpt = None
if args.load_dir is not None:
    ckpt_path = os.path.abspath(args.load_dir)
    if rank == 0:
        lg.info(f'==> Getting ckpt for resuming at {ckpt_path} ...')
    assert os.path.isfile(ckpt_path), '==> Error: no checkpoint file found!'
    loaded_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if rank == 0:
        lg.info(f'==> Getting ckpt for resuming complete.\n')

# Get criterion.
if cfg.get('lb_smooth', 0) > 0:
    criterion = LabelSmoothCELoss(cfg.lb_smooth, cfg.num_classes)
else:
    criterion = torch.nn.CrossEntropyLoss()


def test(net):
    global test_loader
    
    net.eval()
    with torch.no_grad():
        tot_loss, tot_pred, tot_correct, tot_it = 0., 0, 0, len(test_loader)
        for it, (inputs, targets) in enumerate(test_loader):
            if using_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            tot_loss += loss.item()
            _, predicted = outputs.max(1)
            tot_pred += targets.size(0)
            tot_correct += predicted.eq(targets).sum().item()
    
    return tot_loss / tot_it, 100. * tot_correct / tot_pred


def build_model():
    global cfg, loaded_ckpt, lg
    if rank == 0:
        lg.info('==> Building model..')
    net: torch.nn.Module = {
        'FCNet': FCNet,
    }[cfg.model.name](*cfg.model.kwargs)
    init_params(net)
    
    if loaded_ckpt is not None:
        net.load_state_dict(loaded_ckpt['model'])
    num_para = sum(p.numel() for p in net.parameters()) / 1e6
    if rank == 0:
        lg.info(f'==> Building model complete, type: {type(net)}, param:{num_para} * 10^6.\n')
    return net.cuda() if using_gpu else net


def build_op(net):
    global cfg, loaded_ckpt

    if cfg.optm.get('nowd', False):
        nowd_dict = {
            'conv_b': {'weight_decay': 0.0},
            'linear_b': {'weight_decay': 0.0},
            'bn_w': {'weight_decay': 0.0},
            'bn_b': {'weight_decay': 0.0},
        }
        if 'nowd_dict' in cfg.optm:
            nowd_dict.update(cfg.optm['nowd_dict'])
        cfg.optm.kwargs.params, _ = param_group_all(model=net, nowd_dict=nowd_dict)
    else:
        cfg.optm.kwargs.params = net.parameters()

    cfg.optm.kwargs.lr = cfg.optm.base_lr
    op = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': AdamW,
    }[cfg.optm.name](*cfg.optm.kwargs)
    if loaded_ckpt is not None:
        op.load_state_dict(loaded_ckpt['optimizer'])
    return op


def build_sche(optimizer, start_epoch, T_max):
    global cfg

    if cfg.optm.sche == 'cos':
        return CosineLRScheduler(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=cfg.optm.min_lr,
            base_lr=cfg.optm.base_lr,
            warmup_lr=cfg.optm.lr,
            warmup_steps=max(round(cfg.epochs * 0.04), 1),
            last_iter=start_epoch - 1
        )
    elif cfg.optm.sche == 'con':
        return ConstScheduler(
            lr=cfg.optm.lr
        )
    else:
        raise AttributeError(f'unknown scheduler type: {cfg.optm.sche}')


def main():
    global rank, args, loaded_ckpt, lg
    
    # Initialize.
    net: torch.nn.Module = build_model()
    optimizer = build_op(net)
    scheduler = build_sche(optimizer, start_epoch=0, T_max=args.epochs * len(train_loader))
    
    # if args.resume_path:
    #     if rank == 0:
    #         lg.info('==> Reading loaded_ckpt ...')
    #     start_loop = loaded_ckpt['last_loop'] + 1
    #     baseline = loaded_ckpt['baseline']
    #     initial_test_loss = loaded_ckpt['initial_test_loss']
    #     initial_test_acc = loaded_ckpt['initial_test_acc']
    #     avg_mean_best_acc_tensor = loaded_ckpt['avg_mean_best_acc_tensor']
    #     agent.load_state(loaded_ckpt['agent'])
    #     if rank == 0:
    #         lg.info(f'==> Reading loaded_ckpt complete,'
    #                 f' initial test loss: {initial_test_loss:.4f}, acc: {initial_test_acc:3f},'
    #                 f' reward baseline: {baseline}',
    #                 f' avg_mean_best_acc_tensor: {avg_mean_best_acc_tensor},'
    #                 f' start at loop[{start_loop}].\n')
    #         [summary_tb_lg.add_scalars('avg_mean_best_accs', {'baseline': baseline}, t) for t in [0, args.loops / 2, args.loops - 1]]

    # scheduler = build_sche(optimizer, start_epoch=0)
    tot_time, best_acc = 0, 0
    train_loss_avg = AverageMeter(cfg.tb_lg_freq)
    train_acc_avg = AverageMeter(cfg.tb_lg_freq)
    speed_avg = AverageMeter(0)
    for epoch in range(cfg.epochs):
        scheduler.step()
        
        # train a epoch
        tot_it = len(train_loader)
        last_t = time.time()
        for it, (inputs, targets) in enumerate(train_loader):
            data_t = time.time()
            if using_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
    
            optimizer.zero_grad()
            outputs = net(inputs)
            # outputs = outputs[0]
            loss = criterion(outputs, targets)
            loss.backward()
            train_loss_avg.update(loss.item())
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            optimizer.step()
            train_t = time.time()
    
            _, predicted = outputs.max(1)
            pred, correct = targets.size(0), predicted.eq(targets).sum().item() # targets.size(0) i.e. batch_size(or tail batch size)
            train_acc_avg.update(val=100. * correct / pred, num=pred / cfg.batch_size)
            
            if (it % cfg.tb_lg_freq == 0 or it == tot_it - 1) and rank == 0:
                tb_lg.add_scalar('train_loss', train_loss_avg.avg, it + tot_it * epoch)
                tb_lg.add_scalar('train_acc', train_acc_avg.avg, it + tot_it * epoch)
                tb_lg.add_scalar('lr', scheduler.get_lr()[0], it + tot_it * epoch)

            if it % cfg.val_freq == 0 or it == tot_it - 1:
                test_loss, test_acc = test(net)
                net.train()
                if rank == 0:
                    remain_secs = (tot_it - it - 1) * speed_avg.avg + tot_it * (cfg.epochs - epoch - 1) * speed_avg.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    lg.info(
                        f'ep[{epoch}/{cfg.epochs}], it[{it + 1}/{tot_it}]:'
                        f' t-acc[{train_acc_avg.val:.3f}] ({train_acc_avg.avg:.3f}),'
                        f' t-loss[{train_loss_avg.val:.4f}] ({train_loss_avg.avg:.4f}),'
                        f' v-acc[{test_acc:.3f}],'
                        f' v-loss[{test_loss:.4f}]'
                        f' data[{data_t - last_t:.3f}], train[{train_t - data_t:.3f}]'
                        f' rem-t[{remain_time}] ({finish_time})'
                    )
                    tb_lg.add_scalar('test_loss', test_loss, it + tot_it * epoch)
                    tb_lg.add_scalar('test_acc', test_acc, it + tot_it * epoch)

                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)
                if rank == 0 and is_best:
                    model_ckpt_path = os.path.join(args.save_dir, f'best.pth.tar')
                    lg.info(f'==> Saving best model ckpt (epoch[{epoch}], acc{test_acc:.3f}) at {os.path.abspath(model_ckpt_path)}...')
                    torch.save({
                        'model': net.state_dict()
                    }, model_ckpt_path)
                    lg.info(f'==> Saving best model complete.')

            speed_avg.update(time.time() - last_t)
            last_t = time.time()

        if rank == 0 and cfg.save_many:
            model_ckpt_path = os.path.join(args.save_dir, f'ckpt_ep{epoch}.pth.tar')
            lg.info(f'==> Saving model ckpt (epoch[{epoch}], acc{test_acc:.3f}) at {os.path.abspath(model_ckpt_path)}...')
            torch.save({
                'model': net.state_dict()
            }, model_ckpt_path)
            lg.info(f'==> Saving model at epoch{epoch} complete.')
        
    if rank == 0:
        lg.info(
            f'==> End training,'
            f' total time cost: {tot_time:.3f},'
            f' best test acc: {best_acc:.3f}'
        )
        tb_lg.close()
        # 'last_loop': loop,
        # 'baseline': baseline,
        # 'initial_test_loss': initial_test_loss,
        # 'initial_test_acc': initial_test_acc,
        # 'avg_mean_best_acc_tensor': avg_mean_best_acc_tensor,
        # 'model': net.state_dict(),
        # 'agent': agent.state_dict(),
        # 'optimizer': optimizer.state_dict()
    
    # link.finalize()


def plt_data():
    global test_set, lg
    

if __name__ == '__main__':
    if args.only_plt:
        plt_data()
    elif args.only_val:
        pass
    else:
        main()
