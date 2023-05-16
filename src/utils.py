import os
import sys
from glob import glob
import json
import time
import shutil

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