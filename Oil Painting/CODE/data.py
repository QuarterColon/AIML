import os
import re
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


framObjTrain = {'img' : [],
                'mask' : []
                }

framObjVal = {'img' : [],
                'mask' : []
                }


def LoadData(frameObj = None, imgPath = None, maskPath = None, shape=256):
    imgNames = os.listdir(imgPath)
    maskNames = []

    for mem in imgNames:
        maskNames.append(re.sub('\.png', '_L.png', mem))

    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'

    for i in range (len(imgNames)):
        img = plt.imread(imgAddr + imgNames[i])
        mask = plt.imread(maskAddr + maskNames[i])

        img = cv2.resize(img, (shape,shape))
        mask = cv2.resize(mask, (shape, shape))

        frameObj['img'].append(img)
        frameObj['mask'].append(mask)

    return frameObj

if __name__ == "__main__":
    framObjTrain = LoadData(framObjTrain, imgPath='D:/Ankit_Chakraborty/Final_TASK/CamVid/train', maskPath='D:/Ankit_Chakraborty/Final_TASK/CamVid/train_labels', shape=256)
    framObjVal = LoadData(framObjVal, imgPath='D:/Ankit_Chakraborty/Final_TASK/CamVid/test', maskPath='D:/Ankit_Chakraborty/Final_TASK/CamVid/test_labels', shape=256)
    print(np.array(framObjTrain['img']).shape)



