import os
import time
import datetime
import torch

from models import *
from optimizer import AdamW
from loss import LabelSmoothCELoss
from scheduler import CosineLRScheduler, ConstScheduler
from utils import param_group_all, init_params, AverageMeter


class ClsSolver(object):
    def __init__(self, train_loader, test_loader, args, cfg, lg, tb_lg):
        self.args, self.cfg = args, cfg
        self.train_loader, self.test_loader = train_loader, test_loader
        self.lg, self.tb_lg = lg, tb_lg
        if self.cfg.get('lb_smooth', 0) > 0:
            self.criterion = LabelSmoothCELoss(cfg.lb_smooth, cfg.model.kwargs.input_size)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.net: torch.nn.Module = self.build_model()
        self.optm = self.build_op(self.net)
        self.sche = self.build_sche(self.optm, start_epoch=0, T_max=cfg.epochs * len(train_loader))
        


    def test(self):
        self.net.eval()
        with torch.no_grad():
            tot_loss, tot_pred, tot_correct, tot_it = 0., 0, 0, len(self.test_loader)
            for it, (inputs, targets) in enumerate(self.test_loader):
                if self.cfg.using_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
            
                tot_loss += loss.item()
                _, predicted = outputs.max(1)
                tot_pred += targets.size(0)
                tot_correct += predicted.eq(targets).sum().item()
    
        return tot_loss / tot_it, 100. * tot_correct / tot_pred

    def build_model(self):
        self.lg.info('==> Building model..')
        net: torch.nn.Module = {
            'FCNet': FCNet,
        }[self.cfg.model.name](**self.cfg.model.kwargs)
        init_params(net)
    
        num_para = sum(p.numel() for p in net.parameters()) / 1e6
        self.lg.info(f'==> Building model complete, type: {type(net)}, param:{num_para} * 10^6.\n')
        return net.cuda() if self.cfg.using_gpu else net

    def build_op(self, net):
        if self.cfg.optm.get('nowd', False):
            nowd_dict = {
                'conv_b': {'weight_decay': 0.0},
                'linear_b': {'weight_decay': 0.0},
                'bn_w': {'weight_decay': 0.0},
                'bn_b': {'weight_decay': 0.0},
            }
            if 'nowd_dict' in self.cfg.optm:
                nowd_dict.update(self.cfg.optm['nowd_dict'])
            self.cfg.optm.kwargs.params, _ = param_group_all(model=net, nowd_dict=nowd_dict)
        else:
            self.cfg.optm.kwargs.params = net.parameters()
    
        self.cfg.optm.kwargs.lr = self.cfg.optm.base_lr
        op = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamw': AdamW,
        }[self.cfg.optm.name](**self.cfg.optm.kwargs)
        return op

    def build_sche(self, optm, start_epoch, T_max):
        if self.cfg.optm.sche == 'cos':
            return CosineLRScheduler(
                optimizer=optm,
                T_max=T_max,
                eta_min=self.cfg.optm.min_lr,
                base_lr=self.cfg.optm.base_lr,
                warmup_lr=self.cfg.optm.lr,
                warmup_steps=max(round(T_max * 0.04), 1),
                last_iter=start_epoch - 1
            )
        elif self.cfg.optm.sche == 'con':
            return ConstScheduler(
                lr=self.cfg.optm.lr
            )
        else:
            raise AttributeError(f'unknown scheduler type: {self.cfg.optm.sche}')

    def train(self):
        start_train_t = time.time()
        best_acc = 0
        train_loss_avg = AverageMeter(self.cfg.tb_lg_freq)
        train_acc_avg = AverageMeter(self.cfg.tb_lg_freq)
        speed_avg = AverageMeter(0)
        test_loss_avg = AverageMeter(4 * self.cfg.val_freq)
        test_acc_avg = AverageMeter(4 * self.cfg.val_freq)
        for epoch in range(self.cfg.epochs):
        
            # train a epoch
            tot_it = len(self.train_loader)
            last_t = time.time()
            for it, (inputs, targets) in enumerate(self.train_loader):
                data_t = time.time()
                self.sche.step()
                if self.cfg.using_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()

                self.optm.zero_grad()
                outputs = self.net(inputs)
                # outputs = outputs[0]
                loss = self.criterion(outputs, targets)
                loss.backward()
                train_loss_avg.update(loss.item())
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.optm.step()
                train_t = time.time()
            
                _, predicted = outputs.max(1)
                pred, correct = targets.size(0), predicted.eq(targets).sum().item()  # targets.size(0) i.e. batch_size(or tail batch size)
                train_acc_avg.update(val=100. * correct / pred, num=pred / self.cfg.batch_size)
            
                if it % self.cfg.tb_lg_freq == 0 or it == tot_it - 1:
                    self.tb_lg.add_scalar('train_loss', train_loss_avg.avg, it + tot_it * epoch)
                    self.tb_lg.add_scalar('train_acc', train_acc_avg.avg, it + tot_it * epoch)
                    self.tb_lg.add_scalar('lr', self.sche.get_lr()[0], it + tot_it * epoch)
            
                if it % self.cfg.val_freq == 0 or it == tot_it - 1:
                    test_loss, test_acc = self.test()
                    self.net.train()
                    test_loss_avg.update(test_loss)
                    test_acc_avg.update(test_acc)

                    remain_secs = (tot_it - it - 1) * speed_avg.avg + tot_it * (self.cfg.epochs - epoch - 1) * speed_avg.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    self.lg.info(
                        f'ep[{epoch}/{self.cfg.epochs}], it[{it + 1}/{tot_it}]:'
                        f' t-acc[{train_acc_avg.val:.3f}] ({train_acc_avg.avg:.3f}),'
                        f' t-loss[{train_loss_avg.val:.4f}] ({train_loss_avg.avg:.4f}),'
                        f' v-acc[{test_acc_avg.val:.3f}] ({test_acc_avg.avg:.3f}),'
                        f' v-loss[{test_loss_avg.val:.4f}] ({test_loss_avg.avg:.4f}),'
                        f' data[{data_t - last_t:.3f}], train[{train_t - data_t:.3f}]'
                        f' lr[{self.sche.get_lr()[0]:.4g}]'
                        f' rem-t[{remain_time}] ({finish_time})'
                    )
                    self.tb_lg.add_scalar('test_loss', test_loss, it + tot_it * epoch)
                    self.tb_lg.add_scalar('test_acc', test_acc, it + tot_it * epoch)
                
                    is_best = test_acc > best_acc
                    best_acc = max(test_acc, best_acc)
                    if is_best:
                        model_ckpt_path = os.path.join(self.args.save_dir, f'best.pth.tar')
                        self.lg.info(f'==> Saving best model ckpt (epoch[{epoch}], acc{test_acc:.3f}) at {os.path.abspath(model_ckpt_path)}...')
                        torch.save({
                            'model': self.net.state_dict()
                        }, model_ckpt_path)
                        self.lg.info(f'==> Saving best model complete.')
            
                speed_avg.update(time.time() - last_t)
                last_t = time.time()

            if self.cfg.save_many:
                model_ckpt_path = os.path.join(self.args.save_dir, f'ckpt_ep{epoch}.pth.tar')
                self.lg.info(f'==> Saving model ckpt (epoch[{epoch}], acc{test_acc_avg.avg:.3f}) at {os.path.abspath(model_ckpt_path)}...')
                torch.save({
                    'model': self.net.state_dict()
                }, model_ckpt_path)
                self.lg.info(f'==> Saving model at epoch{epoch} complete.')
    
        self.lg.info(
            f'==> End training,'
            f' total time cost: {time.time() - start_train_t:.3f},'
            f' best test acc: {best_acc:.3f}'
        )
        self.tb_lg.close()
