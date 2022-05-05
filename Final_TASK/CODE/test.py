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



model = tf.keras.models.load_model("model.h5")


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
    target_size=(128,128)
)

test_generator = zip(image_generator_test, mask_generator_test)

def test_generator_fn():
    for(img, mask) in test_generator:
        new_mask = adjust_mask(mask)
        yield (img, new_mask)


def visualize_seg(img, gt_mask, shape='normal', gt_mode='sparse'):
  plt.figure(1)
  
  # Img
  plt.subplot(311)
  plt.imshow(img)
  
  # Predict
  pred_mask = model.predict(np.expand_dims(img, 0))
  pred_mask = np.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[0]
  if shape=='flat':
    pred_mask = np.reshape(pred_mask, (256,256)) # Reshape only if you use the flat model. O.w. you dont need
  
  rgb_mask = np.apply_along_axis(map_class_to_rgb, -1, np.expand_dims(pred_mask, -1))
  
  # Prediction
  plt.subplot(312)
  plt.imshow(rgb_mask)
              
  # GT mask
  if gt_mode == 'ohe':
    gt_img_ohe = np.argmax(gt_mask, axis=-1)
    gt_mask = np.apply_along_axis(map_class_to_rgb, -1, np.expand_dims(gt_img_ohe, -1))              
  
  plt.subplot(313)
  plt.imshow((gt_mask).astype(np.uint8))
  plt.show()

# def predict16 (valMap, model, shape = 256):
#     ## getting and proccessing val data
#     img = valMap[0]
#     mask = valMap[1]
#     mask = mask[0:16]
    
#     imgProc = img [0:16]
#     imgProc = np.array(img)
    
#     predictions = model.predict(imgProc)
#     for i in range(len(predictions)):
#         predictions[i] = cv2.merge((predictions[i,:,:,0],predictions[i,:,:,1],predictions[i,:,:,2]))
    
#     return predictions, imgProc, mask

# def Plotter(img, predMask, groundTruth):
#     plt.figure(figsize=(7,7))
    
#     plt.subplot(1,3,1)
#     plt.imshow(img)
#     plt.title('image')
    
#     plt.subplot(1,3,2)
#     plt.imshow(predMask)
#     plt.title('Predicted Mask')
    
#     plt.subplot(1,3,3)
#     plt.imshow(groundTruth)
#     plt.title('actual Mask')

# X_test, y_test = test_generator_fn()
# prediction, actuals, mask = predict16(test_generator_fn(), model)

# Plotter(actuals[1], prediction, mask[1])



visualize_seg(next(image_generator_test)[0],next(mask_generator_test)[0],gt_mode= 'ohe')
