import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader         # Dataset and DataLoader will be used below
import torchvision.transforms as transforms
import random
from label_train_test_video import *
from skimage import io, color, exposure

class spatial_dataset(Dataset):  # inherit from SuperClass: Dataset
    def __init__(self, dic, root_dir, mode, transform=None):

        print '-----------------init_spatial_dataset-------------------'
        self.keys = dic.keys()
        self.values = dic.values()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_UCF_image(self,video_name, index):  # read an image and change it to specific format (pytorch desire)

        if index < 100 and index > 9:  # **************************************  new Add by me
            path = self.root_dir + 'v_' + video_name + '/frame0000'
        elif index > 99 and index < 1000:
            path = self.root_dir + 'v_' + video_name + '/frame000'
        elif index < 10:
            path = self.root_dir + 'v_' + video_name + '/frame00000'
        else:
            path = self.root_dir + 'v_' + video_name + '/frame00'

        img = Image.open(path +str(index)+'.jpg') #<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=342x256 at 0x7F19DCF3D7D0>
     #   print img.mode    # RGB
     #   print img.size    # 342x256
     #   print img.format  # JPEG
        transformed_img = self.transform(img)           # [torch.FloatTensor of size 3x224x224]
        img.close()

        return transformed_img

    def __getitem__(self, idx):
        if self.mode == 'train':
            video_name, nb_frames = self.keys[idx].split(' ')   #Swing_g09_c02    116
            nb_frames = int(nb_frames)
            index_frames = []

            # random select frames
        #    index_frames.append(random.randint(1, nb_frames/3))
        #    index_frames.append(random.randint(nb_frames/3, nb_frames*2/3))
        #    index_frames.append(random.randint(nb_frames*2/3, nb_frames))

            # determined selected frames
            index_frames.append((1 + nb_frames / 3) / 2)
            index_frames.append((nb_frames / 3 + nb_frames * 2 / 3) / 2)
            index_frames.append((nb_frames * 2 / 3 + nb_frames) / 2)

        elif self.mode == 'val':
            video_name, index = self.keys[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1                   # Note:label will start from 0 instead of 1

        #data and sample
        if self.mode=='train':
            data ={}
            for i in range(len(index_frames)):
                key = 'img'+str(i)
                index = index_frames[i]
                data[key] = self.load_UCF_image(video_name, index) #data:{'img2':[3x224x224],'img1':[3x224x224],'img0':[3x224x224]}

            sample = (data, label)   #sample: ({'img2':[3x224x224],'img1':[3x224x224],'img0':[3x224x224]}, 5)

        elif self.mode=='val':
            data = self.load_UCF_image(video_name,index)
            sample = (video_name, data, label)  # sample: ('MoppingFloor_g04_c01', [3x224x224], 54)

        else:
            raise ValueError('There are only train and val mode')

        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, UCF_list):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count ={}
        # Step 1: Obtain Training and Test videos and the corresponding labels
        labeler = UCF_label(path=UCF_list)
        self.train_video, self.test_video = labeler.label_video()
        # train_video={'HandStandPushups_g23_c04': 37, 'MilitaryParade_g17_c05': 53, ...}

    def load_frame_count(self):  # Step 2: Obtain video name and corresponding frame_count
        print '==> Loading frame number of each video'
        with open('/media/ming/DATADRIVE1/UCF101 Multi-stream Code/UCF_list/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)  # {'v_Lunges_g07_c01.avi': 248, 'v_Haircut_g18_c04.avi': 263, ...

        file.close()

        for line in dic_frame :                                 # v_Lunges_g07_c01.avi
            videoname = line.split('_', 1)[1].split('.', 1)[0]  # Lunges_g07_c01
            n, g = videoname.split('_', 1)
            if n == 'HandStandPushups':            # actually no 'HandStandPushups' but 'HandstandPushups' in dic_frame
                videoname = 'HandstandPushups_' + g
            self.frame_count[videoname] = dic_frame[line]  # preserve video name and corresponding frame_count in a dictionary form
    #    print self.frame_count  #{'PommelHorse_g01_c03': 373, 'PommelHorse_g01_c02': 400, ..., 'PoleVault_g02_c02': 79}


    def run(self):
        self.load_frame_count()       # Step 2: Obtain video name and corresponding frame_count
        self.get_training_dic()       # Step 3: Generate frame numbers of each training video
        self.val_sample20()           # Step 4: Uniformly sampling 19 frames at equal interval for each test video
        train_loader = self.train()   # Step 5: Package training data into pytorch data format
        val_loader = self.validate()  # Step 6: Package test data into pytorch data format

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):     # Step 3: Generate frame numbers of each training video
        print '==> Generate frame numbers of each training video'
        self.dic_training={}
        for video in self.train_video:              # MilitaryParade_g17_c04
            nb_frame = self.frame_count[video]
            key = video+' '+ str(nb_frame)          # Billiards_g09_c04  277      Note:here 277 is the total number
            self.dic_training[key] = self.train_video[video]  # {'Billiards_g09_c04 277': 12, ...}  here 277 denotes the number of frames


    def val_sample20(self):  # Step 4: Sampling testing frames, uniformly sample 19 frames at equal interval in each video
        print '==> sampling testing frames'
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]
            interval = int(nb_frame/19)
            for i in range(19):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video] # {'Billiards_g09_c04 1': 12,'Billiards_g09_c04 20': 12,...} here 1,20,...denote sampled frames

        #  print len(self.dic_testing)         # 71877 = 3783 * 19


    def train(self):
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print '==> Training data :',len(training_set),'frames'   # 9537
        print training_set[1][0]['img1'].size()                  # (3L, 224L, 224L)
        print training_set[0][0]['img1'].size()                  # (3L, 224L, 224L)
    #    print training_set[1][0].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        print len(train_loader)                                  # 954

        return train_loader

    def validate(self):
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print '==> Validation data :',len(validation_set),'frames'    # 71877
        print validation_set[1][1].size()                             # (3L, 224L, 224L)
        print '........................v size.................'
    #    print validation_set[1000000000][100000000000].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        print len(val_loader)                                    #7188
    #    print validation_set[1000000000][100000000000].size()


        return val_loader


if __name__ == '__main__':
    
    dataloader = spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                path='/media/ming/DATADRIVE1/UCF101new/UCF-101/',
                                ucf_list='/media/ming/DATADRIVE1/two stream/two-stream-code/UCF_list/',
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()