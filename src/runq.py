import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str)
args = parser.parse_args()
direct = args.directory

params_files = [os.path.join(direct, f) for f in os.listdir(direct) if os.path.isfile(os.path.join(direct, f)) and f.endswith('.json')]

for i, params_path in enumerate(params_files):
    print(f'{i+1} of {len(params_files)}')
    subprocess.run(['py', 'train.py', '-p', params_path])
    subprocess.run(['py', 'test.py', '-l'])