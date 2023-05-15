import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
from unet import build_unet
from utils import load_json, timestr, create_dir, copy_file_to, save_summary
from keras_unet_collection import models
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
parser.add_argument('-n', '--number', type=int)
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
    """
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        name: prefix of the created keras model and its layers.
    """
    model1 = models.unet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', output_activation='Softmax', 
                           batch_norm=True, pool='max', unpool='nearest')
    
    """    
        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone. 
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.
    """
    
    selection = args.number
    model = None
    if selection == None:
        model = models.transunet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                                stack_num_down=2, stack_num_up=2,
                                embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                                activation='ReLU', mlp_activation='ReLU', output_activation='Softmax', 
                                batch_norm=True, pool='max', unpool='nearest')
    elif selection == 0: # WORKS
        model = models.att_unet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes,
                           stack_num_down=2, stack_num_up=2,
                           activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Softmax', 
                           batch_norm=True, pool='max', unpool='nearest')
    elif selection == 1: # WORKS
        model = models.unet_plus_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes,
                            stack_num_down=2, stack_num_up=2,
                            activation='ReLU', output_activation='Softmax',
                            batch_norm=True, pool='max', unpool='nearest', deep_supervision=False)
        
    elif selection == 2: # CRASHES
        model = models.unet_3plus_2d(input_shape, filter_num_down=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                             filter_num_skip='auto', filter_num_aggregate='auto', 
                             stack_num_down=2, stack_num_up=2, 
                             activation='ReLU', output_activation='Softmax',
                             batch_norm=True, pool='max', unpool='nearest', deep_supervision=False)
    elif selection == 3: # WORKS
        model = models.r2_unet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                          stack_num_down=2, stack_num_up=2, recur_num=2,
                          activation='ReLU', output_activation='Softmax', 
                          batch_norm=True, pool='max', unpool='nearest')
    elif selection == 4: # CRASHES
        model = models.resunet_a_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                            dilation_num=[1, 3, 15, 31], 
                            aspp_num_down=256, aspp_num_up=128, 
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool='max', unpool='nearest')
    elif selection == 5: # CRASHES
        model = models.transunet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                                stack_num_down=2, stack_num_up=2,
                                embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                                activation='ReLU', mlp_activation='ReLU', output_activation='Softmax', 
                                batch_norm=True, pool='max', unpool='nearest')
    elif selection == 6: # WORKS
        model = models.unet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', output_activation='Softmax', 
                           batch_norm=True, pool='max', unpool='nearest',
                           backbone='ResNet101V2', weights='imagenet', 
                           freeze_backbone=False, freeze_batch_norm=False)
    elif selection == 7: # WORKS
        model = models.unet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', output_activation='Softmax', 
                           batch_norm=True, pool='max', unpool='nearest',
                           backbone='DenseNet121', weights='imagenet', 
                           freeze_backbone=False, freeze_batch_norm=False)
    elif selection == 8: # WORKS
        model = models.unet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', output_activation='Softmax', 
                           batch_norm=True, pool='max', unpool='nearest',
                           backbone='ResNet101', weights='imagenet', 
                           freeze_backbone=False, freeze_batch_norm=False)
    elif selection == 9: # WORKS
        model = models.unet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', output_activation='Softmax', 
                           batch_norm=True, pool='max', unpool='nearest',
                           backbone='ResNet101', weights=None, 
                           freeze_backbone=False, freeze_batch_norm=False)
    elif selection == 10: # WORKS
        model = models.unet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', output_activation='Softmax', 
                           batch_norm=True, pool='max', unpool='nearest',
                           backbone='VGG16', weights='imagenet', 
                           freeze_backbone=False, freeze_batch_norm=False)

    """
    wip1 = models.swin_unet_2d((128, 128, 3), filter_num_begin=64, n_labels=3, depth=4, stack_num_down=2, stack_num_up=2, 
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
                            output_activation='Softmax', shift_window=True, name='swin_unet')
    wip2 = models.u2net_2d((128, 128, 3), n_labels=2, 
                        filter_num_down=[64, 128, 256, 512], filter_num_up=[64, 64, 128, 256], 
                        filter_mid_num_down=[32, 32, 64, 128], filter_mid_num_up=[16, 32, 64, 128], 
                        filter_4f_num=[512, 512], filter_4f_mid_num=[256, 256], 
                        activation='ReLU', output_activation=None, 
                        batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net')
    wip3 = models.vnet_2d(input_shape, filter_num=[64, 128, 256, 512, 1024], n_labels=num_classes,
                      res_num_ini=1, res_num_max=3, 
                      activation='ReLU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest')
    """
    metrics= [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=metrics
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
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
        TensorBoard(log_dir=os.path.join(build_path, "buildlog"))
    ]

    with tf.device('/GPU:0'):
        save_summary(build_path, model)
        copy_file_to(params_path, os.path.join(build_path, "params.json"))
        model.fit(train_ds,
            validation_data=valid_ds,
            epochs=num_epochs,
            callbacks=callbacks
        )

