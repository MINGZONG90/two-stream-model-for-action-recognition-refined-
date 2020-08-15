from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
import dataloader

rgb_preds='record/spatial/spatial_video_preds_best_300.pickle'
saliency_preds = 'record/saliency/saliency_video_preds_best_300.pickle'

with open(rgb_preds,'rb') as f:
    rgb =pickle.load(f)
f.close()
with open(saliency_preds,'rb') as f:
    saliency =pickle.load(f)
f.close()

dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1,
                                           path='/media/ming/DATADRIVE1/KTH Dataset 600/KTH1saliency/',
                                           KTH_list='/media/ming/DATADRIVE1/KTH Multi-stream Code/KTH_list/'
                                           )
train_loader,val_loader,test_video = dataloader.run()

video_level_preds = np.zeros((len(rgb.keys()),101))
video_level_labels = np.zeros(len(rgb.keys()))
ii=0
for name in sorted(rgb.keys()):
    r = rgb[name]
    o = saliency[name]

    label = int(test_video[name])-1

    video_level_preds[ii,:] = (r+o)
    video_level_labels[ii] = label
    ii+=1


video_level_labels = torch.from_numpy(video_level_labels).long()
video_level_preds = torch.from_numpy(video_level_preds).float()
        
top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))
                                
print top1,top5  #[torch.FloatTensor of size 1]
