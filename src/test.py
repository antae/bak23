import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score, jaccard_score
from utils import load_json, timestr, create_dir, find_latest_model
from tkinter import filedialog, Tk
import argparse
import segmentation_models as sm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

global image_h
global image_w
global num_classes
global classes
global rgb_codes

def load_testset(path, dataset_params):
    test_size = dataset_params["test_size"]

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "test", "labels", "*.png")))

    if test_size != "All" and test_size >= 0 and test_size < len(test_x):
        test_x = test_x[:test_size]
        test_y = test_y[:test_size]

    return (test_x, test_y)

def input_model_path(initialdir):
    Tk().withdraw()
    return filedialog.askopenfilename(initialdir=initialdir, filetypes= (("model files","*.h5"), ("all files","*.*")))

def grayscale_to_3d(image):
    h, w = image.shape[0], image.shape[1]
    output = []

    for pixel in image.flatten():
        output.append([pixel, pixel, pixel])

    output = np.reshape(output, (h, w, 3))
    return output

def grayscale_to_rgb(mask, rgb_codes):
    h, w = mask.shape[0], mask.shape[1]
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(rgb_codes[pixel])

    output = np.reshape(output, (h, w, 3))
    return output

def save_results(image_x, mask, pred_uncert, pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = grayscale_to_rgb(mask, rgb_codes)

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, rgb_codes)

    pred_uncert = np.expand_dims(pred_uncert, axis=-1)
    pred_uncert = grayscale_to_3d(pred_uncert)

    vert_line = np.ones((image_x.shape[0], 10, 3)) * 255
    hor_line = np.ones((10, image_x.shape[0]*2 + 10, 3)) * 255

    cat_vert_images1 = np.concatenate([image_x, vert_line, mask], axis=1)
    cat_vert_images2 = np.concatenate([pred_uncert, vert_line, pred], axis=1)
    cat_images = np.concatenate([cat_vert_images1, hor_line, cat_vert_images2], axis=0)

    cv2.imwrite(save_image_path, cat_images)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default=None)
parser.add_argument('-l', '--latest', action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    config = load_json("config.json")

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Paths """
    dataset_path = config["dataset_path"]
    models_path = config["models_path"]
    latest_path = find_latest_model(models_path) if args.latest else None
    model_path = latest_path if latest_path else args.model if args.model != None and os.path.exists(args.model) else input_model_path(models_path)
    
    build_path = os.path.split(model_path)[0]
    test_path = os.path.join(build_path, 'test' + timestr())
    results_path = os.path.join(test_path, 'results')
    bad_results_path = os.path.join(test_path, 'bad_results')
    score_path = os.path.join(test_path, 'score.csv')
    cm_path = os.path.join(test_path, 'confusion_matrix.png')

    create_dir(test_path)
    create_dir(bad_results_path)
    create_dir(results_path)

    """ Hyperparameters """
    params = load_json(os.path.join(build_path, "params.json"))
    image_h = params["image_h"]
    image_w = params["image_w"]
    num_classes = len(config["classes"])

    """ RGB Code and Classes """
    rgb_codes = config["rgb_codes"]
    classes = config["classes"]

    """ Loading the dataset """
    (test_x, test_y) = load_testset(dataset_path, params["dataset"])
    print(f"Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Load the model """

    custom_objects = {"iou_score": sm.metrics.IOUScore, "f1-score": sm.metrics.FScore, 
                      "focal_loss_plus_dice_loss": sm.losses.categorical_focal_dice_loss,
                      "focal_loss": sm.losses.categorical_focal_loss,
                      "focal_loss_plus_jaccard_loss": sm.losses.categorical_focal_jaccard_loss}
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)

    """ Prediction & Evaluation """
    SCORE = []
    labels = [i for i in range(num_classes)]
    bad_mask_count = 0
    test_results_size = params["dataset"]["test_results_size"]
    test_results_size = test_results_size if test_results_size != "All" else len(test_x)
    cumulative_cm = None
    for i, [x, y] in enumerate(tqdm(zip(test_x, test_y), total=len(test_x))):
        """ Extract the name """
        name = x.split("\\")[-1].split(".")[0]
        #print(name)
        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image_w, image_h))
        image_x = image
        image = image/255.0 ## (H, W, 3)
        image = np.expand_dims(image, axis=0) ## [1, H, W, 3]
        image = image.astype(np.float32)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int32)

        """ Prediction """
        pred = model.predict(image, verbose=0)[0]
        pred_uncert = np.amax(pred, axis=-1) * 255
        pred = np.argmax(pred, axis=-1)
        pred = pred.astype(np.int32)

        """ Save the results """
        if (i < test_results_size):
            save_image_path = os.path.join(results_path, name + '.png')
            save_results(image_x, mask, pred_uncert, pred, save_image_path)

        """ Flatten the array """
        flat_mask = mask.flatten()
        flat_pred = pred.flatten()

        cm = confusion_matrix(flat_mask, flat_pred, labels=labels)
        if cumulative_cm is None:
            cumulative_cm = cm
        else:
            cumulative_cm = np.add(cumulative_cm, cm)

        """ Calculating the metrics values """
        f1_value = f1_score(flat_mask, flat_pred, labels=labels, average=None, zero_division=0)
        jac_value = jaccard_score(flat_mask, flat_pred, labels=labels, average=None, zero_division=0)

        if bad_mask_count < 100: 
            f1_mean = np.mean(f1_value)
            if f1_mean < 0.55:
                save_image_path = os.path.join(bad_results_path, name + '.png')
                save_results(image_x, mask, pred_uncert, pred, save_image_path)
                bad_mask_count += 1
        
        SCORE.append([f1_value, jac_value])

    score = np.array(SCORE)
    score = np.mean(score, axis=0)

    cumulative_cm = cumulative_cm.astype(float)
    for row in cumulative_cm:
        s = sum(row)
        for i in range(len(row)):
            row[i] = round(row[i] / s, 2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cumulative_cm, display_labels=classes)
    disp.plot() 
    disp.ax_.set(xlabel='Spėjamos klasės', ylabel='Klasės')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    f = open(score_path, "w", encoding='utf-8')
    f.write("Class,F1,Jaccard\n")

    l = ["Class", "F1", "Jaccard"]
    print(f"{l[0]:15s} {l[1]:10s} {l[2]:10s}")
    print("-"*35)

    for i in range(num_classes):
        class_name = classes[i]
        f1 = score[0, i]
        jac = score[1, i]
        dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
        print(dstr)
        f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

    print("-"*35)
    class_mean = np.mean(score, axis=-1)
    class_name = "Mean"

    f1 = class_mean[0]
    jac = class_mean[1]

    dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
    print(dstr)
    f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

    f.close()
