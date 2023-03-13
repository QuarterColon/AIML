from sklearn import metrics
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from gc import callbacks
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from segmentation_models import Unet
import segmentation_models as sm
from segmentation_models.metrics import iou_score
from data_AUG import train_generator_fn, val_generator_fn
sm.set_framework('tf.keras')
sm.framework()

from data import LoadData
from model import build_unet



if __name__ == "__main__":


    """HYPERPARAMETERS"""
    shape = (128,128, 3)
    num_classes = 32
    lr = 1e-4
    batch_size = 16
    epochs = 200

    """MODEL"""
    
    model = Unet(backbone_name='inceptionresnetv2', encoder_weights='imagenet', encoder_freeze=True, activation="softmax", classes =32)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4), loss="categorical_crossentropy", metrics = ['accuracy'])
    callbacks = [
        ModelCheckpoint("model.h5", verbose=1, save_best_model=True)
        ]
    
    model.fit(train_generator_fn(),epochs= epochs,callbacks=callbacks,
        validation_data=val_generator_fn(),
        validation_steps=100//16,
        steps_per_epoch=369//16)   

    