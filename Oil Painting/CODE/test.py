import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from data_AUG import map_class_to_rgb, adjust_mask

from data import LoadData
from model import build_unet



model = tf.keras.models.load_model("D:\Ankit_Chakraborty\Final_TASK\model.h5")


"""TEST DATA"""
seed = 1
batch_size = 16
data_path="D:/Ankit_Chakraborty/Final_TASK/CamVid/"
data_gen_args_test = dict(rescale = 1./255)

mask_gen_args_test = dict()

image_datagen_test = ImageDataGenerator(**data_gen_args_test)
mask_datagen_test = ImageDataGenerator(**mask_gen_args_test)

image_generator_test = image_datagen_test.flow_from_directory(data_path,
    class_mode=None,
    classes=['val'],
    seed=seed,
    batch_size=batch_size,
    target_size=(128,128))

mask_generator_test = mask_datagen_test.flow_from_directory(
    data_path,
    class_mode=None,
    classes=['val_labels'],
    seed=seed,
    batch_size=batch_size,
    color_mode= 'rgb',
    target_size=(128,128)
)

test_generator = zip(image_generator_test, mask_generator_test)

def test_generator_fn():
    for(img, mask) in test_generator:
        new_mask = adjust_mask(mask)
        yield (img, new_mask)


def visualize_seg(img, gt_mask, shape='normal', gt_mode=''):
  plt.figure(1)
  
  
  plt.subplot(311)
  plt.imshow(img)
  
  
  pred_mask = model.predict(np.expand_dims(img, 0))
  pred_mask = np.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[0]
  if shape=='flat':
    pred_mask = np.reshape(pred_mask, (256,256)) 
  
  rgb_mask = np.apply_along_axis(map_class_to_rgb, -1, np.expand_dims(pred_mask, -1))
  
  plt.subplot(312)
  plt.imshow(rgb_mask)
              
  if gt_mode == 'ohe':
    gt_img_ohe = np.argmax(gt_mask, axis=-1)
    gt_mask = np.apply_along_axis(map_class_to_rgb, -1, np.expand_dims(gt_img_ohe, -1))              
  
  plt.subplot(313)
  plt.imshow((gt_mask).astype(np.uint8))
  plt.show()


visualize_seg(next(image_generator_test)[10],next(mask_generator_test)[10])

