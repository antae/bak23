import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str)
args = parser.parse_args()
direct = args.directory


for i in range(5):
    path = os.path.join(direct, f"params{i}.json")
    subprocess.run(['py', 'train.py', '-p', path, '--number', str(i)])

for i in range(6, 11):
    path = os.path.join(direct, f"params{i}.json")
    subprocess.run(['py', 'train.py', '-p', path, '--number', str(i)])
