from mnist import MNIST
import numpy as np
import cv2
import os

def save_images(ims, labels, save_dir):
    for i in range(len(labels)):
        img = np.array(ims[i]).reshape(28, 28)
        label = labels[i]

        savepath = os.path.join(save_dir, str(label))
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        cv2.imwrite(os.path.join(savepath, str(i)+".png"), img)
        print savepath + "\t" + str(label)
    pass

mndata = MNIST('./data')
train_ims, train_labels = mndata.load_training()
test_ims, test_labels = mndata.load_testing()
#print type(train_ims), type(train_labels)
#print type(test_ims),type(test_labels)

grayImage = np.array(test_ims[0]).reshape(28, 28)
label = test_labels[0]

save_test_dir = "/home/zhangxin/github/python-mnist/data/test/"
save_train_dir = "/home/zhangxin/github/python-mnist/data/train/"

save_images(train_ims, train_labels, save_train_dir)
    #print savepath
    #print img
    #print img.shape
    #print label
    #break