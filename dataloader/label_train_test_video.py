import os, pickle


class UCF_label():
    def __init__(self, path):
        self.path = path

    def get_action_index(self): # Obtain class name and corresponding label (char) because testlist01.txt has no labels
        self.action_label={}
        with open(self.path+'classInd.txt') as f: #/media/ming/DATADRIVE1/UCF101 Multi-stream Code/UCF_list/classInd.txt
            content = f.readlines()  #return list according to lines: ['1 ApplyEyeMakeup\n', '2 ApplyLipstick\n', ..., '100 WritingOnBoard\n', '101 YoYo\n']
            content = [x.strip('\r\n') for x in content]  #remove the specified character in the head and tail of the string: ['1 ApplyEyeMakeup', '2 ApplyLipstick', ...,'100 WritingOnBoard', '101 YoYo']
        f.close()
        for line in content:
            label,action = line.split(' ')  #slice a string by specifying a delimeter
            if action not in self.action_label.keys():
                self.action_label[action]=label
      #  print self.action_label   #{'MilitaryParade': '53', 'TrampolineJumping': '94', ...,'Haircut': '34', 'TennisSwing': '92'}

    def label_video(self):  # Obtain Training and Test videos and the corresponding labels
        self.get_action_index()
        for path,subdir,files in os.walk(self.path):
        #    print path      # /media/ming/DATADRIVE1/UCF101 Multi-stream Code/UCF_list/
        #    print subdir    # []
        #    print files     # ['classInd.txt', 'frame_count.txt', 'testlist.txt', 'trainlist.txt']
            for filename in files:
                if filename.split('.')[0] == 'trainlist01':
                    print '-----------------get_train_video_labels-----------------------------'
                    train_video = self.file2_dic(self.path+filename) # train_video={'HandStandPushups_g23_c04': 37, 'MilitaryParade_g17_c05': 53, ...}

                if filename.split('.')[0] == 'testlist01':
                    print '-----------------get_test_video_labels------------------------------'
                    test_video = self.file2_dic(self.path+filename)

        print '==> (Training video, Validation video):(', len(train_video),len(test_video),')' #(9537 3783) 9537+3783=13320
        train_video2 = self.name_HandstandPushups(train_video)
        test_video2 = self.name_HandstandPushups(test_video)
        return train_video2, test_video2

    def file2_dic(self,fname):     #Make all videos in the database have labels (int)
        print fname
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic={}
        for line in content: # ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1 just for trainlist01.txt, testlist01.txt doesn't has labels
            key = line.split('/',1)[1].split(' ',1)[0].split('_',1)[1].split('.',1)[0]  # ApplyEyeMakeup_g08_c01
            label = self.action_label[line.split('/')[0]]  # 1
            dic[key] = int(label)  # char -> int
        return dic

    #lower(S):S->s   HandstandPushups/v_HandStandPushups_g01_c01.avi -> HandstandPushups/v_HandstandPushups_g01_c01.avi
    def name_HandstandPushups(self,dic):
        dic2 = {}
        for video in dic:
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            else:
                videoname=video
            dic2[videoname] = dic[video]
        return dic2


if __name__ == '__main__':

    path = '../KTH_list/'
    labeler = UCF_label(path=path)
    train_video2,test_video2 = labeler.label_video()
    print len(train_video2),len(test_video2)