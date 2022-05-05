from tkinter import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




data = pd.read_csv('D:\Ankit_Chakraborty\Final_TASK\CamVid\class_dict.csv', index_col=0)
cls2rgb = {cl:list(data.loc[cl, :]) for cl in data.index}
idx2rgb = {idx:np.array(rgb) for idx, (cl, rgb) in enumerate(cls2rgb.items())}

def map_class_to_rgb(p):
  
  return idx2rgb[p[0]]

def adjust_mask(mask, flat=False):

    semantic_map = []
    for colour in list(cls2rgb.values()):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)

    semantic_map = np.stack(semantic_map, axis=-1)
    if flat:
        semantic_map = np.reshape(semantic_map, (-1,128*128))
        
    return np.float32(semantic_map)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    rescale = 1./255)

mask_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**mask_gen_args)

seed = 1

data_path = 'D:/Ankit_Chakraborty/Final_TASK/CamVid/'
batch_size = 16
image_generator = image_datagen.flow_from_directory(
    data_path,
    class_mode=None,
    classes=['train'],
    seed=seed,
    batch_size = batch_size,
    target_size=(128,128))

mask_generator = mask_datagen.flow_from_directory(data_path,
    class_mode=None,
    classes=['train_labels'],
    seed=seed,
    batch_size = batch_size,
    color_mode= 'rgb',
    target_size=(128,128))

train_generator = zip(image_generator, mask_generator)

def train_generator_fn():
    for (img,mask) in train_generator:
        new_mask = adjust_mask(mask)
        yield (img, new_mask)

data_gen_args_val = dict(rescale = 1./255)

mask_gen_args_val = dict()

image_datagen_val = ImageDataGenerator(**data_gen_args_val)
mask_datagen_val = ImageDataGenerator(**mask_gen_args_val)

image_generator_val = image_datagen_val.flow_from_directory(data_path,
    class_mode=None,
    classes=['val'],
    seed=seed,
    batch_size=batch_size,
    target_size=(128,128))

mask_generator_val = mask_datagen_val.flow_from_directory(
    data_path,
    class_mode=None,
    classes=['val_labels'],
    seed=seed,
    batch_size=batch_size,
    color_mode='rgb',
    target_size=(128,128)
)

val_generator = zip(image_generator_val, mask_generator_val)

def val_generator_fn():
    for(img, mask) in val_generator:
        new_mask = adjust_mask(mask)
        yield (img, new_mask)




