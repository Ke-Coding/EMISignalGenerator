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

from models import *
from optimizer import AdamW
from loss import LabelSmoothCELoss
from emi_dataset import EMIDataset
from utils import create_exp_dir, create_logger, set_seed, param_group_all, init_params, AverageMeter


parser = argparse.ArgumentParser(description='EMI signals classification')
parser.add_argument('--log_dir', type=str, default='exp')
parser.add_argument('--input_size', required=True, type=int)
parser.add_argument('--num_classes', required=True, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--optm', default='adam', type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=0.0001, type=float)
parser.add_argument('--nowd', action='store_true', default=False)
parser.add_argument('--dropout_p', default=0, type=float)
parser.add_argument('--tb_lg_freq', default=8, type=int)
parser.add_argument('--val_freq', default=64, type=int)
parser.add_argument('--data_path', default='', type=str)
parser.add_argument('--save_many', action='store_true', default=False)

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--lb_smooth', default=0.1, type=float)
parser.add_argument('--grad_clip', default=5, type=float)


# Parsing args
args = parser.parse_args()

# rank, world_size = dist_init()
# local_rank, __, self_group_name, leader_group_name = group_split(
#     global_rank=rank, world_size=world_size,
#     num_groups=args.num_groups, group_size=args.group_size)

rank = 0

if rank == 0:
    print(f'==> Raw args: {args}')
args.optm = args.optm.strip().lower()
# args.base_lr = args.lr if args.sche == 'con' else args.lr / 4 # todo: 加sche别忘了把build_op那里改成args.base_lr

run_time = time.strftime("%Y%m%d-%H%M%S")
time_ascii_tensor = torch.from_numpy(np.array([ord(c) for c in run_time]))
# link.broadcast(time_ascii_tensor, 0)
sync_run_time = ''.join([chr(c.item()) for c in time_ascii_tensor])

args.exp_time = sync_run_time
args.log_dir = f'{args.log_dir}-{args.exp_time}'
args.save_dir = os.path.join(args.log_dir, f'total_ckpt')

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
    tb_lg.add_text('exp_time', args.exp_time)
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

if args.data_path is None or len(args.data_path) == 0:
    raise AttributeError(f'data file {args.data_path} not found!')

if rank == 0:
    lg.info(f'==> Reading dataset from {args.data_path} ...')

train_set = EMIDataset(
    fpath=args.data_path, train=True, num_classes=args.num_classes, normalize=True)
test_set = EMIDataset(
    fpath=args.data_path, train=False, num_classes=args.num_classes, normalize=True)

if rank == 0:
    lg.info(f'==> Getting dataloader from {args.data_path} ...')
train_loader = DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)
test_loader = DataLoader(
    dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

classes = ('pink', 'brown', 'laplace', 'uniform', 'exponential')
assert args.num_classes == len(classes)
if rank == 0:
    lg.info(f'==> Preparing data complete, classes:{classes} .\n')

# Load checkpoints.
resumed_checkpoint = None
# if args.resume_path:
#     fpath = os.path.abspath(args.resume_path)
#     if rank == 0:
#         lg.info(f'==> Getting ckpt for resuming at {fpath} ...')
#     assert os.path.isfile(fpath), '==> Error: no checkpoint file found!'
#     resumed_checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
#     if rank == 0:
#         lg.info(f'==> Getting ckpt for resuming complete.\n')

if args.lb_smooth > 0:
    criterion = LabelSmoothCELoss(args.lb_smooth, args.num_classes)
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
    global args, resumed_checkpoint, lg
    if rank == 0:
        lg.info('==> Building model..')
    net: torch.nn.Module = FCNet(input_dim=args.input_size, output_dim=args.num_classes,
                                 hid_dims=[256, 192, 128, 72], dropout_p=args.dropout_p,
                                 af_name='swish')
    init_params(net)
    
    # if args.resume_path:
    #     net.load_state_dict(resumed_checkpoint['model'])
    num_para = sum(p.numel() for p in net.parameters()) / 1e6
    if rank == 0:
        lg.info(f'==> Building model complete, type: {type(net)}, param:{num_para} * 10^6.\n')
    return net.cuda() if using_gpu else net


def build_op(net):
    global args, resumed_checkpoint
    if args.nowd:
        op_params, _ = param_group_all(model=net, nowd_dict={
            'conv_b': {'weight_decay': 0.0},
            'linear_b': {'weight_decay': 0.0},
            'bn_w': {'weight_decay': 0.0},
            'bn_b': {'weight_decay': 0.0},
        })
    else:
        op_params = net.parameters()
    op = None
    if args.optm == 'sgd':
        op = torch.optim.SGD(params=op_params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optm == 'adam':
        op = AdamW(params=op_params, lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    # if args.resume_path:
    #     op.load_state_dict(resumed_checkpoint['optimizer'])
    return op


# def build_sche(optimizer, start_epoch):
#     global args
#
#     if args.sche == 'cos':
#         return CosineLRScheduler(
#             optimizer=optimizer,
#             T_max=args.epochs,
#             eta_min=args.learning_rate_min,
#             base_lr=args.base_lr,
#             warmup_lr=args.lr,
#             warmup_steps=max(round(args.epochs * 0.04), 1),
#             last_iter=start_epoch - 1
#         )
#     elif args.sche == 'con':
#         return ConstScheduler(
#             lr=args.lr
#         )
#     else:
#         raise AttributeError(f'unknown scheduler type: {args.sche}')


def main():
    global rank, args, resumed_checkpoint, lg
    
    # Initialize.
    net: torch.nn.Module = build_model()
    optimizer = build_op(net)
    
    # if args.resume_path:
    #     if rank == 0:
    #         lg.info('==> Reading resumed_checkpoint ...')
    #     start_loop = resumed_checkpoint['last_loop'] + 1
    #     baseline = resumed_checkpoint['baseline']
    #     initial_test_loss = resumed_checkpoint['initial_test_loss']
    #     initial_test_acc = resumed_checkpoint['initial_test_acc']
    #     avg_mean_best_acc_tensor = resumed_checkpoint['avg_mean_best_acc_tensor']
    #     agent.load_state(resumed_checkpoint['agent'])
    #     if rank == 0:
    #         lg.info(f'==> Reading resumed_checkpoint complete,'
    #                 f' initial test loss: {initial_test_loss:.4f}, acc: {initial_test_acc:3f},'
    #                 f' reward baseline: {baseline}',
    #                 f' avg_mean_best_acc_tensor: {avg_mean_best_acc_tensor},'
    #                 f' start at loop[{start_loop}].\n')
    #         [summary_tb_lg.add_scalars('avg_mean_best_accs', {'baseline': baseline}, t) for t in [0, args.loops / 2, args.loops - 1]]

    # scheduler = build_sche(optimizer, start_epoch=0)
    tot_time, best_acc = 0, 0
    train_loss_avg = AverageMeter(args.tb_lg_freq)
    train_acc_avg = AverageMeter(args.tb_lg_freq)
    speed_avg = AverageMeter(0)
    for epoch in range(args.epochs):
        # scheduler.step()
        
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            train_t = time.time()
    
            _, predicted = outputs.max(1)
            pred, correct = targets.size(0), predicted.eq(targets).sum().item() # targets.size(0) i.e. batch_size(or tail batch size)
            train_acc_avg.update(val=100. * correct / pred, num=pred / args.batch_size)
            
            if (it % args.tb_lg_freq == 0 or it == tot_it - 1) and rank == 0:
                tb_lg.add_scalar('train_loss', train_loss_avg.avg, epoch)
                tb_lg.add_scalar('train_acc', train_acc_avg.avg, epoch)
                # tb_lg.add_scalar('lr', scheduler.get_lr()[0], epoch)

            if it % args.val_freq == 0 or it == tot_it - 1:
                test_loss, test_acc = test(net)
                net.train()
                if rank == 0:
                    remain_secs = (tot_it - it - 1) * speed_avg.avg + tot_it * (args.epochs - epoch - 1) * speed_avg.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    lg.info(
                        f'ep[{epoch}/{args.epochs}], it[{it + 1}/{tot_it}]:'
                        f' t-acc[{train_acc_avg.val:.3f}] ({train_acc_avg.avg:.3f}),'
                        f' t-loss[{train_loss_avg.val:.4f}] ({train_loss_avg.avg:.4f}),'
                        f' v-acc[{test_acc:.3f}],'
                        f' v-loss[{test_loss:.4f}]'
                        f' data[{data_t - last_t:.3f}], train[{train_t - data_t:.3f}]'
                        f' rem-t[{remain_time}] ({finish_time})'
                    )
                    tb_lg.add_scalar('test_loss', test_loss, epoch)
                    tb_lg.add_scalar('test_acc', test_acc, epoch)

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

        if rank == 0 and args.save_many:
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


if __name__ == '__main__':
    main()
