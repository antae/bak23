import os
import sys
from glob import glob
import json
import time
import shutil

def load_json(path):
    f = open(path)  
    content = json.load(f)
    f.close()
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