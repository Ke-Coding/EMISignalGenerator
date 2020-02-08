import os
import time
import datetime
import torch

from .basic_solver import BasicSolver
from loss import LabelSmoothCELoss
from utils import AverageMeter


class ClsSolver(BasicSolver):
    def __init__(self, train_loader, test_loader, encoder, cfg, lg, tb_lg):
        super(ClsSolver, self).__init__(train_loader, test_loader, cfg, lg, tb_lg)
        
        self.cfg.model.kwargs.encoder = encoder
        if self.cfg.get('lb_smooth', 0) > 1e-7:
            self.criterion = LabelSmoothCELoss(self.cfg.lb_smooth, self.cfg.model.kwargs.input_size)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        
        super(ClsSolver, self).build_solver()
    
    def test_solver(self):
        self.net.eval()
        with torch.no_grad():
            tot_loss, tot_pred, tot_correct, tot_it = 0., 0, 0, len(self.test_loader)
            for it, (inputs, targets) in enumerate(self.test_loader):
                if it == 0:
                    # cnt = 0
                    # for n, p in self.net.encoder.named_parameters():
                    #     print(n, p)
                    #     cnt += 1
                    #     if cnt == 3:
                    #         break
                if self.cfg.using_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
            
                tot_loss += loss.item()
                _, predicted = outputs.max(1)
                tot_pred += targets.size(0)
                tot_correct += predicted.eq(targets).sum().item()
    
        return tot_loss / tot_it, 100. * tot_correct / tot_pred

    def train_solver(self):
        start_train_t = time.time()
        best_acc = 0
        last_iter = -1
        train_loss_avg = AverageMeter(self.cfg.tb_lg_freq)
        train_acc_avg = AverageMeter(self.cfg.tb_lg_freq)
        test_loss, test_acc = 0, 0
        speed_avg = AverageMeter(0)
        for epoch in range(self.cfg.epochs):
        
            # train a epoch
            tot_it = len(self.train_loader)
            last_t = time.time()
            for it, (inputs, targets) in enumerate(self.train_loader):
                data_t = time.time()
                last_iter += 1
                self.sche.step()
                if self.cfg.using_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
            
                self.optm.zero_grad()
                outputs = self.net(inputs)
                # outputs = outputs[0]
                loss = self.criterion(outputs, targets)
                loss.backward()
                train_loss_avg.update(loss.item())
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.net.parameters()), self.cfg.grad_clip)
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
                    test_loss, test_acc = self.test_solver()
                    self.net.train()
                
                    remain_secs = (tot_it - it - 1) * speed_avg.avg + tot_it * (self.cfg.epochs - epoch - 1) * speed_avg.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    self.lg.info(
                        f'ep[{epoch}/{self.cfg.epochs}], it[{it + 1}/{tot_it}]:'
                        f' t-acc[{train_acc_avg.val:.3f}] ({train_acc_avg.avg:.3f}),'
                        f' t-loss[{train_loss_avg.val:.4f}] ({train_loss_avg.avg:.4f}),'
                        f' v-acc[{test_acc:.3f}],'
                        f' v-loss[{test_loss:.4f}],'
                        f' data[{data_t - last_t:.3f}], train[{train_t - data_t:.3f}]'
                        f' lr[{self.sche.get_lr()[0]:.4g}]'
                        f' rem-t[{remain_time}] ({finish_time})'
                    )
                    self.tb_lg.add_scalar('test_loss', test_loss, it + tot_it * epoch)
                    self.tb_lg.add_scalar('test_acc', test_acc, it + tot_it * epoch)
                
                    is_best = test_acc > best_acc
                    best_acc = max(test_acc, best_acc)
                    if is_best:
                        model_ckpt_path = os.path.join(self.cfg.save_dir, f'cls_best.pth.tar')
                        self.lg.info(f'==> Saving cls best model ckpt (epoch[{epoch}], acc{test_acc:.3f}) at {os.path.abspath(model_ckpt_path)}...')
                        torch.save({
                            'model': self.net.state_dict(),
                            'optm': self.optm.state_dict(),
                            'last_iter': last_iter
                        }, model_ckpt_path)
                        self.lg.info(f'==> Saving cls best model complete.')
            
                speed_avg.update(time.time() - last_t)
                last_t = time.time()
        
            if self.cfg.save_many:
                model_ckpt_path = os.path.join(self.cfg.save_dir, f'cls_ep{epoch}.pth.tar')
                self.lg.info(f'==> Saving cls model ckpt (epoch[{epoch}], last_iter[{last_iter}], v-acc: {test_acc:.3f, v-loss: {test_loss:.3f}}) at {os.path.abspath(model_ckpt_path)}...')
                torch.save({
                    'model': self.net.state_dict(),
                    'optm': self.optm.state_dict(),
                    'last_iter': last_iter
                }, model_ckpt_path)
                self.lg.info(f'==> Saving cls model at epoch{epoch} complete.')
    
        self.tb_lg.add_scalar('best_test_acc', best_acc, 0)
        self.tb_lg.add_scalar('best_test_acc', best_acc, len(self.train_loader) * self.cfg.epochs)
        self.lg.info(
            f'==> End training,'
            f' total time cost: {time.time() - start_train_t:.3f},'
            f' best test acc: {best_acc:.3f}'
        )
