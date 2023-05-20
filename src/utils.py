import os
import sys
from glob import glob
import json
import time
import shutil

import cv2
import numpy as np

def load_json(path):
    with open(path, encoding='utf-8') as f:
        content = json.load(f)
    return content

def timestr():
    return time.strftime("%Y%m%d-%H%M%S")

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file_to(src, dst):
    shutil.copy(src, dst)

def save_summary(path, model):
    with open(os.path.join(path, "model.txt"), 'w') as f:
        sys.stdout = f
        model.summary()
        sys.stdout = sys.__stdout__

def find_latest_model(path):
    import os
    subdirs = next(os.walk(path))[1]
    subdirs = [subdir for subdir in subdirs if subdir.startswith('build')]
    sorted_subdirs = sorted(subdirs, reverse=True)
    if sorted_subdirs:
        new_path = os.path.join(os.path.join(path, sorted_subdirs[0]), "model.h5")
        return new_path
    else:
        return None
    

def read_image_and_mask(test_x, test_y, size):

    images = []
    masks = []
    for [x, y] in zip(test_x, test_y):
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (size, size))
        image = image/255.0 ## (H, W, 3)
        image = np.expand_dims(image, axis=0) ## [1, H, W, 3]
        image = image.astype(np.float32)
        images.append(image)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int32)
        mask = mask.flatten()
        masks.append(mask)

    return images, masks

from keras.callbacks import Callback
from sklearn.metrics import f1_score

class MacroF1(Callback):
        def __init__(self, model, inputs, targets, path):
            self.model = model
            self.inputs = inputs
            self.targets = targets
            self.labels = [i for i in range(11)]
            self.path = path

        def on_epoch_end(self, epoch, logs):
            f1_macro_scores = []
            for input, target in zip(self.inputs, self.targets):
                pred = self.model.predict(input, verbose=0)[0]
                pred = np.argmax(pred, axis=-1)
                pred = pred.astype(np.int32)
                pred = pred.flatten()
                f1_macro = f1_score(target, pred, labels=self.labels, average='macro', zero_division=0)
                f1_macro_scores.append(f1_macro)

            f1_mean = np.mean(f1_macro_scores)
            print(f"macro F1Score: {f1_mean:.4f}")
            f = open(self.path, "a", encoding='utf-8')
            f.write(f"{epoch},{f1_mean:.4f}\n")
            f.close()
