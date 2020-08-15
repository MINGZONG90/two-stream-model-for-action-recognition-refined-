import pickle,os
from PIL import Image
import scipy.io
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# other util
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output     [torch.cuda.FloatTensor of size 10x6 (GPU 0)]
    # target     [torch.cuda.LongTensor of size 10 (GPU 0)]

    maxk = max(topk)              # 5
    batch_size = target.size(0)   # 10

    _, pred = output.topk(maxk, 1, True, True)             # pred is index   [torch.cuda.LongTensor of size 10x5 (GPU 0)]
    pred = pred.t()                                        # transpose       [torch.cuda.LongTensor of size 5x10 (GPU 0)]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # the size -1 is inferred from other dimensions

#    print target.view(1, -1)                               # [torch.cuda.LongTensor of size 1x10 (GPU 0)]
#    print target.view(1, -1).expand_as(pred)               # [torch.cuda.LongTensor of size 5x10 (GPU 0)] same with above
#    print correct                                          # [torch.cuda.ByteTensor of size 5x10 (GPU 0)]   0,1 value

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)     # 10
        res.append(correct_k.mul_(100.0 / batch_size))      # [10, 80]
#        print correct[:k]  # k=1: 0     0     0     0     0     0     0     1     1     0

    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#def save_checkpoint(state, checkpoint):
#    torch.save(state, checkpoint)

def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)

def record_info(info,filename,mode):

    if mode =='train':
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Loss','Prec@1','Prec@5']
        
    if mode =='test':
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Loss','Prec@1','Prec@5']

    if mode =='two_stream_fusion':
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Prec@1']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)   


