import os
import logging

import torch.nn as nn
import torch.nn.functional as F
import shutil
import torch
import numpy as np
from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = int(length)
        self.history = []
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reset(self):
        if self.length > 0:
            self.history.clear()
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0
    
    def reduce_update(self, tensor, num=1):
        link.allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            # assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count

    def get_trimmed_mean(self):
        if len(self.history) >= 5:
            trimmed = max(int(self.length * 0.1), 1)
            return np.mean(sorted(self.history)[trimmed:-trimmed])
        else:
            return self.avg

    def state_dict(self):
        return vars(self)

    def load_state(self, state_dict):
        self.__dict__.update(state_dict)

                
def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d): # or SyncBatchNorm2d
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('==> Dir created : {}.'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def count_parameters(net):
    return sum([p.numel() for name, p in net.named_parameters() if "auxiliary" not in name]) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    # random.seed(3)
    # torch.manual_seed(0)
    #if torch.cuda.is_available(): torch.cuda.manual_seed_all(1)
    #torch.backends.cudnn.deterministic = True
    # np.random.seed(2)
    #torch.backends.cudnn.benchmark = False
    #os.environ['PYTHONHASHSEED'] = str(4)

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False      # todo 为什么？
    torch.backends.cudnn.deterministic = True   # todo 为什么？


def create_logger(name, log_file, level=logging.INFO, stream=True):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)6s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)
    return l


def param_group_all(model, nowd_dict):
    pgroup_normal = []
    pgroup = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    names = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    if 'conv_dw_w' in nowd_dict:
        pgroup['conv_dw_w'] = []
        names['conv_dw_w'] = []
    if 'conv_dw_b' in nowd_dict:
        pgroup['conv_dw_b'] = []
        names['conv_dw_b'] = []
    if 'conv_dense_w' in nowd_dict:
        pgroup['conv_dense_w'] = []
        names['conv_dense_w'] = []
    if 'conv_dense_b' in nowd_dict:
        pgroup['conv_dense_b'] = []
        names['conv_dense_b'] = []
    if 'linear_w' in nowd_dict:
        pgroup['linear_w'] = []
        names['linear_w'] = []

    names_all = []
    type2num = defaultdict(lambda: 0)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                if 'conv_dw_b' in pgroup and m.groups == m.in_channels:
                    pgroup['conv_dw_b'].append(m.bias)
                    names_all.append(name+'.bias')
                    names['conv_dw_b'].append(name+'.bias')
                    type2num[m.__class__.__name__+'.bias(dw)'] += 1
                elif 'conv_dense_b' in pgroup and m.groups == 1:
                    pgroup['conv_dense_b'].append(m.bias)
                    names_all.append(name+'.bias')
                    names['conv_dense_b'].append(name+'.bias')
                    type2num[m.__class__.__name__+'.bias(dense)'] += 1
                else:
                    pgroup['conv_b'].append(m.bias)
                    names_all.append(name+'.bias')
                    names['conv_b'].append(name+'.bias')
                    type2num[m.__class__.__name__+'.bias'] += 1
            if 'conv_dw_w' in pgroup and m.groups == m.in_channels:
                pgroup['conv_dw_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['conv_dw_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight(dw)'] += 1
            elif 'conv_dense_w' in pgroup and m.groups == 1:
                pgroup['conv_dense_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['conv_dense_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight(dense)'] += 1

        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                pgroup['linear_b'].append(m.bias)
                names_all.append(name+'.bias')
                names['linear_b'].append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1
            if 'linear_w' in pgroup:
                pgroup['linear_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['linear_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight'] += 1
        elif (isinstance(m, torch.nn.BatchNorm2d)
              or isinstance(m, torch.nn.BatchNorm1d)):
              # or isinstance(m, link.nn.SyncBatchNorm2d)):
            if m.weight is not None:
                pgroup['bn_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['bn_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight'] += 1
            if m.bias is not None:
                pgroup['bn_b'].append(m.bias)
                names_all.append(name+'.bias')
                names['bn_b'].append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1

    for name, p in model.named_parameters():
        if name not in names_all:
            pgroup_normal.append(p)

    param_groups = [{'params': pgroup_normal}]
    for ptype in pgroup.keys():
        if ptype in nowd_dict.keys():
            param_groups.append({'params': pgroup[ptype], **nowd_dict[ptype]})
        else:
            param_groups.append({'params': pgroup[ptype]})

        # if logger is not None:
        #     logger.info(ptype)
        #     for k, v in param_groups[-1].items():
        #         if k == 'params':
        #             logger.info('   params: {}'.format(len(v)))
        #         else:
        #             logger.info('   {}: {}'.format(k, v))

    # if logger is not None:
    #     for ptype, pconf in nowd_dict.items():
    #         logger.info('names for {}({}): {}'.format(ptype, len(names[ptype]), names[ptype]))

    return param_groups, type2num


class SwishAutoFn(torch.autograd.Function):
    """ Memory Efficient Swish
    From: https://blog.ceshine.net/post/pytorch-memory-swish/
    """
    
    @staticmethod
    def forward(ctx, x):
        result = x.mul(torch.sigmoid(x))
        ctx.save_for_backward(x)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


def swish(x, inplace=False):
    return SwishAutoFn.apply(x)


# def swish(x, inplace=False):
#     if inplace:
#         return x.mul_(torch.sigmoid(x))
#     else:
#         return x.mul(torch.sigmoid(x))


def hswish(x, inplace=False):
    if inplace:
        return x.mul_(F.relu6(x.add(3.), inplace=True)) / 6.  # 不可以写x.add_，因为前面还要保存x用来乘呢
    else:
        return x * (F.relu6(x.add(3.), inplace=True)) / 6.


def hsigmoid(x, inplace=False):
    if inplace:
        return F.relu6(x.add_(3.), inplace=True) / 6.
    else:
        return F.relu6(x.add(3.), inplace=True) / 6.


def get_af(af_name):
    af_name = af_name.strip().lower()
    af_dic = {
        # 'sigmoid' : F.sigmoid,
        # 'hsigmoid': hsigmoid,
        'relu': F.relu,
        'relu6': F.relu6,
        'swish': swish,
        'hswish': hswish,
    }
    if af_name not in af_dic.keys():
        raise NotImplementedError("activation function not implemented!")
    return af_dic[af_name]

