import os
import time
import datetime
import torch
import torch.nn.functional as F

from .basic_solver import BasicSolver
from loss import vae_loss
from utils import AverageMeter


class VAESolver(BasicSolver):
    def __init__(self, train_loader, test_loader, cfg, lg, tb_lg):
        super(VAESolver, self).__init__(train_loader, test_loader, cfg, lg, tb_lg)
        super(VAESolver, self).build_solver()
        
        # print(self.net)
        
    def get_encoder(self):
        return self.net.get_encoder()
    
    def test_solver(self):
        self.net.eval()
        with torch.no_grad():
            tot_loss, tot_bce, tot_kld, tot_it = 0., 0., 0., len(self.test_loader)
            for it, (inputs, _) in enumerate(self.test_loader):
                if self.cfg.using_gpu:
                    inputs = inputs.cuda()
                if it == 0:
                    gaussian, mu, log_sigma = self.net.encoder(inputs.view(inputs.shape[0], 1, -1))
                    print('mu : ', mu)
                    print('sgm: ', log_sigma)
                    print('gau: ', gaussian)
                x_rec, mu, log_sigma = self.net(inputs)
                bce_loss, kld_loss = vae_loss(inputs, x_rec, mu, log_sigma)
                loss = bce_loss + kld_loss
                tot_loss += loss.item()
                tot_bce += bce_loss.item()
                tot_kld += kld_loss.item()
        
        return tot_loss / tot_it, tot_bce / tot_it, tot_kld / tot_it
    
    def train_solver(self):
        start_train_t = time.time()
        lowest_loss = 1e9
        last_iter = -1
        train_loss_avg = AverageMeter(self.cfg.tb_lg_freq)
        train_bce_avg = AverageMeter(self.cfg.tb_lg_freq)
        train_kld_avg = AverageMeter(self.cfg.tb_lg_freq)
        test_loss = 0
        speed_avg = AverageMeter(0)
        for epoch in range(self.cfg.epochs):
            
            # train a epoch
            tot_it = len(self.train_loader)
            last_t = time.time()
            for it, (inputs, _) in enumerate(self.train_loader):
                data_t = time.time()
                last_iter += 1
                self.sche.step()
                if self.cfg.using_gpu:
                    inputs = inputs.cuda()
                
                self.optm.zero_grad()
                x_rec, mu, log_sigma = self.net(inputs)
                bce_loss, kld_loss = vae_loss(inputs, x_rec, mu, log_sigma)
                loss = bce_loss + kld_loss
                loss.backward()
                train_loss_avg.update(loss.item())
                train_bce_avg.update(bce_loss.item())
                train_kld_avg.update(kld_loss.item())
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.optm.step()
                train_t = time.time()
                
                if it % self.cfg.tb_lg_freq == 0 or it == tot_it - 1:
                    self.tb_lg.add_scalar('train_loss', train_loss_avg.avg, it + tot_it * epoch)
                    self.tb_lg.add_scalar('train_bce', train_bce_avg.avg, it + tot_it * epoch)
                    self.tb_lg.add_scalar('train_kld', train_kld_avg.avg, it + tot_it * epoch)
                    self.tb_lg.add_scalar('lr', self.sche.get_lr()[0], it + tot_it * epoch)
                
                if it % self.cfg.val_freq == 0 or it == tot_it - 1:
                    test_loss, test_bce, test_kld = self.test_solver()
                    self.net.train()
                    
                    remain_secs = (tot_it - it - 1) * speed_avg.avg + tot_it * (self.cfg.epochs - epoch - 1) * speed_avg.avg
                    remain_time = datetime.timedelta(seconds=round(remain_secs))
                    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                    self.lg.info(
                        f'ep[{epoch}/{self.cfg.epochs}], it[{it + 1}/{tot_it}]:'
                        f' t-loss[{train_loss_avg.val:.4f}] ({train_loss_avg.avg:.4f}),'
                        f' t-bce[{train_bce_avg.val:.4f}] ({train_bce_avg.avg:.4f}),'
                        f' t-kld[{train_kld_avg.val:.4f}] ({train_kld_avg.avg:.4f}),'
                        f' v-loss[{test_loss:.4f}],'
                        f' v-bce[{test_bce:.4f}],'
                        f' v-kld[{test_kld:.4f}],'
                        f' data[{data_t - last_t:.3f}], train[{train_t - data_t:.3f}]'
                        f' lr[{self.sche.get_lr()[0]:.4g}]'
                        f' rem-t[{remain_time}] ({finish_time})'
                    )
                    self.tb_lg.add_scalar('test_loss', test_loss, it + tot_it * epoch)
                    self.tb_lg.add_scalar('test_bce', test_bce, it + tot_it * epoch)
                    self.tb_lg.add_scalar('test_kld', test_kld, it + tot_it * epoch)
                    
                    is_best = test_loss < lowest_loss
                    lowest_loss = min(test_loss, lowest_loss)
                    if is_best:
                        model_ckpt_path = os.path.join(self.cfg.save_dir, f'cls_best.pth.tar')
                        self.lg.info(f'==> Saving cls best model ckpt (epoch[{epoch}], loss{lowest_loss:.3f}) at {os.path.abspath(model_ckpt_path)}...')
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
                self.lg.info(f'==> Saving cls model ckpt (epoch[{epoch}], last_iter[{last_iter}], v-loss: {test_loss:.3f}) at {os.path.abspath(model_ckpt_path)}...')
                torch.save({
                    'model': self.net.state_dict(),
                    'optm': self.optm.state_dict(),
                    'last_iter': last_iter
                }, model_ckpt_path)
                self.lg.info(f'==> Saving cls model at epoch{epoch} complete.')
        
        self.tb_lg.add_scalar('lowest_test_loss', lowest_loss, 0)
        self.tb_lg.add_scalar('lowest_test_loss', lowest_loss, len(self.train_loader) * self.cfg.epochs)
        self.lg.info(
            f'==> End training,'
            f' total time cost: {time.time() - start_train_t:.3f},'
            f' lowest test loss: {lowest_loss:.3f}'
        )
