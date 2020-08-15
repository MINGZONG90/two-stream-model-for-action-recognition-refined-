from my_network import *
from utils import *
import torch.optim as optim
import dataloader
import torch
import os

from ResidualNetwork import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Prepare DataLoader
print '*************************Prepare DataLoader********************************'
data_loader = dataloader.Motion_DataLoader(
    BATCH_SIZE=10,
    num_workers=8,
    path='/media/ming/DATADRIVE1/UCF101 Dataset/tvl1_flow/',
    UCF_list='/media/ming/DATADRIVE1/UCF101 Multi-stream Code/UCF_list/',
    in_channel=10                                      # NEW ADD
)

train_loader, test_loader, test_video = data_loader.run()

# build the model
model = resnet50(pretrained= False, channel=10*2).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train_1epoch(epoch):

    print '---------train execute ', epoch , ' epoch'
    loss = 0.0
    # Computes and stores the average and current value
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, sample in enumerate(train_loader):

        data, label = sample  # data=[torch.FloatTensor of size 10x20x224x224]  label=[torch.LongTensor of size 10]
        data_input = Variable(data).cuda()
        label = label.cuda(async=True)  #[torch.LongTensor of size 10]
        target_label = Variable(label).cuda()#Variable containing:70 73 75 48 51 7 12 7 55 75[torch.cuda.LongTensor of size 10 (GPU 0)]

        # compute output
        output = model(data_input)
        loss = criterion(output, target_label)

        # zero the parameter gradients + backward + update the parameter weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1, data.size(0))
        top5.update(prec5, data.size(0))

    # preserve loss and accuracy after an epoch
    info = {'Epoch':[epoch],
            'Loss':[round(losses.avg,5)],
            'Prec@1':[round(top1.avg,4)],
            'Prec@5':[round(top5.avg,4)],
        }
    record_info(info, 'record/opticalflow/opticalflow_train.csv','train')

def test_1epoch(epoch):

    print '---------test execute ', epoch, ' epoch'
    dic_video_level_preds = {}

    for i, sample in enumerate(test_loader):

        # get the inputs
        keys, data, label = sample
    #    print keys            # ('SoccerPenalty_g02_c02', 'MoppingFloor_g04_c01', ...} number=10
    #    print data            # [torch.FloatTensor of size 10x3x224x224]
    #    print label           # 84 54 80 11 84 2 63 4 30 4       [torch.LongTensor of size 10]

        label = label.cuda(async=True)
        data_input = Variable(data, volatile=True).cuda(async=True)    # Variable containing:[torch.cuda.FloatTensor of size 10x3x32x32 (GPU 0)]
     #   frame_level_label = Variable(label, volatile=True).cuda(async=True) # Variable containing:3 1 4 0 5 4 0 5 3 5 [torch.cuda.LongTensor of size 10 (GPU 0)]
        # compute output
        output = model(data_input)          # Variable containing:[torch.cuda.FloatTensor of size 10x6 (GPU 0)]

        # Calculate video level prediction
        preds = output.data.cpu().numpy() # [[-1.16014826 ..., -0.74533045] ... [-0.26081595 ..., 0.15930791]] shape:10*6
        nb_data = preds.shape[0]          # 10

        for j in range(nb_data):
            videoName = keys[j]   # person05_jogging_d3_uncomp = person05_jogging_d3_uncomp

            if videoName not in dic_video_level_preds.keys():
                dic_video_level_preds[videoName] = preds[j, :]
#                print preds[j,:]   # [-0.09742972 -0.01401141 -0.11057711 -0.05899951  0.03018731  0.01602992]

            else:
                dic_video_level_preds[videoName] += preds[j, :]

#    print len(dic_video_level_preds)    # 216

    video_top1, video_top5, video_loss = frame2_video_level_accuracy(dic_video_level_preds)

    info = {'Epoch': [epoch],
            'Loss': [round(video_loss, 5)],
            'Prec@1': [round(video_top1, 4)],
            'Prec@5': [round(video_top5, 4)],
            }
    record_info(info, 'record/opticalflow/opticalflow_test.csv', 'test')

    return video_top1, dic_video_level_preds

def frame2_video_level_accuracy(dic_video_level_preds):

    # initialize
    video_level_preds = np.zeros((len(dic_video_level_preds), 101))  # shape:(216, 6)
    video_level_labels = np.zeros(len(dic_video_level_preds))      # shape:(216,)

    i = 0
    for name in sorted(dic_video_level_preds.keys()):
        preds = dic_video_level_preds[name]
        label = int(test_video[name]) - 1

        video_level_preds[i, :] = preds
        video_level_labels[i] = label
        i += 1

    video_level_preds = torch.from_numpy(video_level_preds).float()
    video_level_labels = torch.from_numpy(video_level_labels).long()

    top1, top5 = accuracy(video_level_preds, video_level_labels, topk=(1, 5))
    loss = criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())

    top1 = float(top1.numpy())
    top5 = float(top5.numpy())

     # print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
    return top1, top5, loss.data.cpu().numpy()


# ************************* Start ****************************
best_prec1 = 0.0
nb_epochs = 1
video_pred_collections = {}

for epoch in range(nb_epochs):    # loop over the dataset multiple times
    print 'train_1epoch---------start'
    train_1epoch(epoch)
    print 'test_1epoch----------start'
    prec1, dic_video_level_preds = test_1epoch(epoch)
    print 'test_1epoch----------stop'
    video_pred_collections[epoch] = dic_video_level_preds

    is_best = prec1 > best_prec1
    if is_best:
        best_prec1 = prec1
        with open('record/opticalflow/opticalflow_video_preds_best.pickle', 'wb') as f:
            pickle.dump(dic_video_level_preds, f)
        f.close()

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1
    }, is_best, 'record/opticalflow/checkpoint.pth', 'record/opticalflow/model_best.pth')

with open('record/opticalflow/opticalflow_video_preds_collections.pickle','wb') as f:
    pickle.dump(video_pred_collections, f)
f.close()

model_best = 'record/opticalflow/model_best.pth'
if os.path.isfile(model_best):
    print "==> loading checkpoint '{}'".format(model_best)
    checkpoint = torch.load(model_best)
    best_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    print best_epoch
    print best_prec1



#print prec1
#with open('record/spatial/spatial_video_preds.pickle','wb') as f:
#    pickle.dump(dic_video_level_preds, f)
#f.close()

#torch.save(model.state_dict(),'record/spatial/checkpoint.pth')



#    save_checkpoint({
#        'epoch': epoch,
#        'state_dict': model.state_dict(),
#    }, 'record/spatial/checkpoint.pth')