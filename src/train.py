import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
from unet import build_unet
from utils import load_json, timestr, create_dir, copy_file_to, save_summary, read_image_and_mask, MacroF1
import argparse
import segmentation_models as sm
import random

global image_h
global image_w
global num_classes

def load_dataset(path, dataset_params):
    train_size = dataset_params["train_size"]
    valid_size = dataset_params["valid_size"]

    #train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")) + glob(os.path.join("E:\\Datasets\\croppedLaPa\\train\\images", "*.jpg")))
    #train_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")) + glob(os.path.join("E:\\Datasets\\croppedLaPa\\train\\labels", "*.png")))
    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")) + glob(os.path.join("E:\\Datasets\\patchedLaPa\\train\\images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")) + glob(os.path.join("E:\\Datasets\\patchedLaPa\\train\\labels", "*.png")))
    #train_x = sorted(glob(os.path.join("E:\\Datasets\\bilateralLaPa", "train", "bilateral_features", "*.jpg")))
    #train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
    #train_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")))
    
    #valid_x = sorted(glob(os.path.join("E:\\Datasets\\bilateralLaPa", "val", "bilateral_features", "*.jpg")))
    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "val", "labels", "*.png")))

    if train_size != "All" and train_size >= 0 and train_size < len(train_x):
        train_x = train_x[:train_size]
        train_y = train_y[:train_size]
    if valid_size != "All" and valid_size >= 0 and valid_size < len(valid_x):
        valid_x = valid_x[:valid_size]
        valid_y = valid_y[:valid_size]

    return (train_x, train_y), (valid_x, valid_y)

def add_noise(image):
    noise = np.random.randint(0,50,(image_h, image_w))
    zitter = np.zeros_like(image)
    zitter[:,:,1] = noise  

    image = cv2.add(image, zitter)
    return image

def read_image_mask_and_add_noise(x, y):
    """ Image """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    x = add_noise(x)
    x = x/255.0
    x = x.astype(np.float32)

    """ Mask """
    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    y = y.astype(np.int32)

    return x, y

def add_transform(image, mask):
    rand_num = random.randint(0, 9)
    if rand_num < 7:
        return image, mask
    
    rand_num = random.randint(0, 1)
    if rand_num == 1:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    rand_num = random.randint(0, 3)
    rot = cv2.ROTATE_90_COUNTERCLOCKWISE
    if rand_num == 1:
        rot = cv2.ROTATE_180
    elif rand_num == 2:
        rot = cv2.ROTATE_90_CLOCKWISE

    if rand_num != 3:
        image = cv2.rotate(image, rot)
        mask = cv2.rotate(mask, rot)

    return image, mask

def read_image_mask(x, y):
    """ Image """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    x = x/255.0
    x = x.astype(np.float32)

    """ Mask """
    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    y = y.astype(np.int32)

    return x, y

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        image, mask = read_image_mask_and_add_noise(x, y)
        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, num_classes)

    image.set_shape([image_h, image_w, 3])
    mask.set_shape([image_h, image_w, num_classes])

    return image, mask

def val_preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        image, mask = read_image_mask(x, y)
        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, num_classes)

    image.set_shape([image_h, image_w, 3])
    mask.set_shape([image_h, image_w, num_classes])

    return image, mask

def tf_dataset(X, Y, batch=8, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    prep = preprocess if training else val_preprocess
    ds = ds.shuffle(buffer_size=5000).map(prep)
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
    train_ds = tf_dataset(train_x, train_y, batch=batch_size, training=True)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size, training=False)

    """ Model """
    metrics = [sm.metrics.IOUScore(threshold=None), sm.metrics.FScore(threshold=None)]
    model = build_unet(input_shape, num_classes)
    model.compile(
        loss=sm.losses.categorical_focal_dice_loss,
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=metrics
    )

    """ Training """
    val_inputs, val_targets = read_image_and_mask(valid_x[:300], valid_y[:300], image_h)
    lr_params = params["reduce_lr"]
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', 
                          factor=lr_params["factor"], 
                          patience=lr_params["patience"], 
                          min_lr=lr_params["min_lr"], 
                          verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
        TensorBoard(log_dir=os.path.join(build_path, "buildlog")),
        MacroF1(model, val_inputs, val_targets, os.path.join(build_path, "macrof1.csv"))
    ]

    with tf.device('/GPU:0'):
        save_summary(build_path, model)
        copy_file_to(params_path, os.path.join(build_path, "params.json"))
        model.fit(train_ds,
            validation_data=valid_ds,
            epochs=num_epochs,
            callbacks=callbacks
        )
        model.save(os.path.join(build_path, "final_model.h5"))

