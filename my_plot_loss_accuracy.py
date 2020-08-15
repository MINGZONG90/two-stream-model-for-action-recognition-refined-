import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

#mpl.rcParams['xtick.labelsize'] = 24
#mpl.rcParams['ytick.labelsize'] = 24

#plt.plot([2,3,4,5,1,6])
#plt.ylabel("Grade")
#plt.xlabel("number")
#plt.axis([-1,11,0,7])
#plt.show()

rgb_data = pd.read_csv('./record/spatial/rgb_test_300.csv')
rgb_epoch = rgb_data['Epoch']
rgb_loss = rgb_data['Loss']
rgb_accuracy = rgb_data['Prec@1']

saliency_data = pd.read_csv('./record/saliency/saliency_test_300.csv')
saliency_epoch = saliency_data['Epoch']
saliency_loss = saliency_data['Loss']
saliency_accuracy = saliency_data['Prec@1']

twostream_data = pd.read_csv('./record/two_stream_fusion/two_stream_fusion_test_300.csv')
twostream_epoch = twostream_data['Epoch']
twostream_accuracy = twostream_data['Prec@1']

plt.title("KTH Test Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuray(%)")
#plt.ylabel("Loss")

#plt.plot(rgb_epoch, rgb_accuracy, 'b', saliency_epoch, saliency_accuracy, 'k', twostream_epoch, twostream_accuracy, 'r')

plt.plot(rgb_epoch, rgb_accuracy, 'b', label = "rgb")
plt.plot(saliency_epoch, saliency_accuracy, 'k', label = "saliency")
plt.plot(twostream_epoch, twostream_accuracy, 'r', label = "fusion")
plt.legend(loc = 0, ncol = 3)

#plt.plot(rgb_epoch, rgb_loss, 'b', label = "rgb")
#plt.plot(saliency_epoch, saliency_loss, 'k', label = "saliency")
#plt.legend(loc = 0, ncol = 2)

plt.savefig("./result.png")
plt.show()

#x = epoch
#y = loss
#z = accuracy
#plt.plot(x, y)  # point diagram: plt.plot(x, y, '.')
#plt.plot(x, z, 'r')
#plt.show()