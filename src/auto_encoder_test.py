import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, Model

SIZE=224
img_data=[]

img=cv2.imread('E:\\Datasets\\LaPa\\train\\images\\68459447_0.jpg')   #Change 1 to 0 for grey images
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Changing BGR to RGB to show images in true colors
img=cv2.resize(img,(SIZE, SIZE))
img_data.append(img_to_array(img))

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.

model = tf.keras.models.load_model("E:\\MLModels\\bak23_2\\build20230519-034425\\final_model.h5")

pred = model.predict(img_array)

#pred_u8 = (pred[0].reshape(128,128,3)).astype(np.uint8)

plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(pred[0].reshape(SIZE,SIZE,3))
plt.title('Reconstructed')
plt.show()