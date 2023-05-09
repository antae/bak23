import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from unet import build_unet
from utils import load_json, timestr, create_dir, copy_file_to, save_summary
import argparse

global image_h
global image_w
global num_classes
global classes
global rgb_codes

def load_dataset(path, dataset_params):
    train_size = dataset_params["train_size"]
    valid_size = dataset_params["valid_size"]

    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")))
    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "val", "labels", "*.png")))

    if train_size != "All" and train_size >= 0 and train_size < len(train_x):
        train_x = train_x[:train_size]
        train_y = train_y[:train_size]
    if valid_size != "All" and valid_size >= 0 and valid_size < len(valid_x):
        valid_x = valid_x[:valid_size]
        valid_y = valid_y[:valid_size]

    return (train_x, train_y), (valid_x, valid_y)

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

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--params', type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    config = load_json("config.json")

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
    params_path = args.params if args.params and os.path.exists(args.params) else "params.json"
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
    (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path, params["dataset"])
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)}")
    print("")

    """ Dataset Pipeline """
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unet(input_shape, num_classes)
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(lr)
    )

    """ Training """
    lr_params = params["reduce_lr"]
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', 
                          factor=lr_params["factor"], 
                          patience=lr_params["patience"], 
                          min_lr=lr_params["min_lr"], 
                          verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    with tf.device('/GPU:0'):
        save_summary(build_path, model)
        copy_file_to(params_path, os.path.join(build_path, "params.json"))
        model.fit(train_ds,
            validation_data=valid_ds,
            epochs=num_epochs,
            callbacks=callbacks
        )

