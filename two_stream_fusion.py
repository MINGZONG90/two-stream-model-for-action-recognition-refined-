from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
import dataloader

rgb_video_pred_collections = 'record/spatial/spatial_video_preds_collections_300.pickle'
saliency_video_pred_collections = 'record/saliency/saliency_video_preds_collections_300.pickle'

with open(rgb_video_pred_collections, 'rb') as f:
    rgbs = pickle.load(f)
f.close()
with open(saliency_video_pred_collections, 'rb') as f:
    saliencys = pickle.load(f)
f.close()

dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1,
                                           path='/media/ming/DATADRIVE1/KTH Dataset 600/KTH1saliency/',
                                           KTH_list='/media/ming/DATADRIVE1/KTH Multi-stream Code/KTH_list/'
                                           )
train_loader, val_loader, test_video = dataloader.run()

nb_epochs  = len(rgbs)

for epoch in range(nb_epochs):

    rgb = rgbs[epoch]
    saliency = saliencys[epoch]

    video_level_preds = np.zeros((len(rgb.keys()), 101))
    video_level_labels = np.zeros(len(rgb.keys()))
    ii = 0
    for name in sorted(rgb.keys()):

        r = rgb[name]
        o = saliency[name]

        label = int(test_video[name]) - 1

        video_level_preds[ii, :] = (r + o)
        video_level_labels[ii] = label
        ii += 1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()
    top1, top5 = accuracy(video_level_preds, video_level_labels, topk=(1, 5))

    info = {'Epoch': [epoch],
            'Prec@1': [round(top1, 4)],
            }
    record_info(info, 'record/two_stream_fusion/two_stream_fusion_test.csv', 'two_stream_fusion')