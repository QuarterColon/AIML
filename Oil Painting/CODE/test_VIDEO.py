import tensorflow as tf
from tensorflow import image
import os
import cv2
import imutils
import numpy as np

net = tf.keras.models.load_model("D:/Ankit_Chakraborty/model.h5")
net.summary()
capture = cv2.VideoCapture('D:/Ankit_Chakraborty/Final_TASK/testvideo.mp4')
writer = None
resize = (128, 128)
frames = []
while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    
    frame = np.array(cv2.resize(frame, resize))
    frames.append(frame)
    frames = np.asarray(frames) 
    output = net.predict(frames)[0]
    cv2.imshow("Frame", output)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break


    
capture.release()
cv2.destroyAllWindows()
