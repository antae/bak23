import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from unet import build_unet
import time
import json

global image_h
global image_w
global num_classes
global classes
global rgb_codes

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):

    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))[:200]
    train_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")))[:200]

    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.jpg")))[:100]
    valid_y = sorted(glob(os.path.join(path, "val", "labels", "*.png")))[:100]

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))[:100]
    test_y = sorted(glob(os.path.join(path, "test", "labels", "*.png")))[:100]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image_mask(x, y):
    """ Image """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    x = x/255.0
    x = x.astype(np.float32)

    """ Mask """
    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (image_w, image_h))
    y = y.astype(np.int32)

    return x, y

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        return read_image_mask(x, y)

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, num_classes)

    image.set_shape([image_h, image_w, 3])
    mask.set_shape([image_h, image_w, num_classes])

    return image, mask

def tf_dataset(X, Y, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(buffer_size=5000).map(preprocess)
    ds = ds.batch(batch).prefetch(2)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Hyperparameters """
    image_h = 224
    image_w = 224
    num_classes = 11
    input_shape = (image_h, image_w, 3)
    batch_size = 4
    lr = 1e-4 ## 0.0001
    num_epochs = 5

    f = open('config.json')  
    config = json.load(f)
    f.close()
    
    """ Paths """
    dataset_path = config["dataset_path"] 
    models_path = config["models_path"]
    model_path_head = os.path.join(models_path, 'build' + time.strftime("%Y%m%d-%H%M%S"))
    create_dir(model_path_head)
    model_path = os.path.join(model_path_head, "model.h5")
    csv_path = os.path.join(model_path_head, "data.csv")

    """ RGB Code and Classes """
    rgb_codes = config["rgb_codes"]
    classes = config["classes"]
    
    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Dataset Pipeline """
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unet(input_shape, num_classes)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )

    """ Training """
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    with tf.device('/GPU:0'):
        model.fit(train_ds,
            validation_data=valid_ds,
            epochs=num_epochs,
            callbacks=callbacks
        )

