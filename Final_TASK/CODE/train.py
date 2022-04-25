from gc import callbacks
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

from data import LoadData
from model import build_unet


if __name__ == "__main__":


    """DATASET"""
    framObjTrain = {'img' : [],
                'mask' : []
                }

    framObjVal = {'img' : [],
                'mask' : []
                }


    framObjTrain = LoadData(framObjTrain, imgPath='D:/Ankit_Chakraborty/Final_TASK/CamVid/train', maskPath='D:/Ankit_Chakraborty/Final_TASK/CamVid/train_labels', shape=256)
    framObjVal = LoadData(framObjVal, imgPath='D:/Ankit_Chakraborty/Final_TASK/CamVid/test', maskPath='D:/Ankit_Chakraborty/Final_TASK/CamVid/test_labels', shape=256)

    """SEEDING"""
    np.random.seed(42)
    tf.random.set_seed(42)

    """HYPERPARAMETERS"""
    shape = (256, 256, 3)
    num_classes = 3
    lr = 1e-4
    batch_size = 16
    epochs = 20

    """MODEL"""
    model = build_unet(shape, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr))

    train_steps = len(framObjTrain)//batch_size
    val_steps = len(framObjVal)//batch_size

    callbacks = [
        ModelCheckpoint("model.h5, verbose=1, save_best_model=True"),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    ]

    model.fit(x=np.array(framObjTrain['img']), y=np.array(framObjTrain['mask']),epochs= epochs,
                steps_per_epoch=369//16, 
                validation_data=[np.array(framObjVal['img']),np.array(framObjVal['mask'])],
                validation_steps=100//16,
                callbacks=callbacks)

