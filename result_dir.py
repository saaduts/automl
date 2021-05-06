import os

def make_dir_if_not_exists(dir):
    if os.path.exists(dir):
        print(f'{dir} already exists!')
    else:
        os.makedirs(dir)
