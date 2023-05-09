import subprocess

# Define a list of Python scripts to run in the pipeline
subprocess.run(['py', 'train.py'])
subprocess.run(['py', 'test.py', '-l'])