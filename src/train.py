import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
from unet import build_autoencoder
from utils import load_json, timestr, create_dir, copy_file_to, save_summary
import argparse
import segmentation_models as sm
from keras_unet_collection import models

global image_h
global image_w
global num_classes

def read_image_mask(x, y):
    """ Image """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))

    x = x/255.0
    x = x.astype(np.float32)

    """ Image """
    y = cv2.imread(y, cv2.IMREAD_COLOR)
    y = cv2.resize(y, (image_w, image_h))

    y = y/255.0
    y = y.astype(np.float32)

    return x, y

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        return read_image_mask(x, y)

    image1, image2 = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])

    image1.set_shape([image_h, image_w, 3])
    image2.set_shape([image_h, image_w, 3])

    return image1, image2

def tf_dataset(X, Y, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(buffer_size=5000).map(preprocess)
    ds = ds.batch(batch).prefetch(2)
    return ds

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--params', type=str, default=None)
parser.add_argument('-n', '--number', type=int)
args = parser.parse_args()

if __name__ == "__main__":
    config = load_json("C:\\Projects\\bak23\\src\\config.json")

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
     
    """ Paths """
    dataset_path = config["dataset_path"] 
    models_path = config["models_path"]
    build_path = os.path.join(models_path, 'build' + timestr())
    model_path = os.path.join(build_path, "model.h5")
    csv_path = os.path.join(build_path, "data.csv")

    create_dir(build_path)

    """ Hyperparameters """
    params_path = args.params if args.params and os.path.exists(args.params) else "C:\\Projects\\bak23\\src\\params.json"
    params = load_json(params_path)

    image_h = params["image_h"]
    image_w = params["image_w"]
    num_classes = len(config["classes"])
    input_shape = (image_h, image_w, 3)
    batch_size = params["batch_size"]
    lr = params["lr"]
    num_epochs = params["num_epochs"]
    loss = params["loss"]

    """ RGB Code and Classes """
    rgb_codes = config["rgb_codes"]
    classes = config["classes"]
    
    """ Loading the dataset """
    train_x = sorted(glob(os.path.join("E:\\Datasets\\Celeb\\img_align_celeba\\*.jpg")))
    train_y = sorted(glob(os.path.join("E:\\Datasets\\Celeb\\img_align_celeba\\*.jpg")))
    print(f"Train: {len(train_x)}/{len(train_x)}")
    print("")

    """ Dataset Pipeline """
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)

    selection = args.number
    model = build_autoencoder(input_shape)

    model.compile(
        optimizer='adam', 
        loss='mean_squared_error', 
        metrics=['accuracy']
    )

    """ Training """
    lr_params = params["reduce_lr"]
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=False),
        CSVLogger(csv_path, append=True)
    ]

    with tf.device('/GPU:0'):
        save_summary(build_path, model)
        copy_file_to(params_path, os.path.join(build_path, "params.json"))
        model.fit(train_ds,
            epochs=num_epochs,
            callbacks=callbacks
        )
        model.save(os.path.join(build_path, "final_model.h5"))

