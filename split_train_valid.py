import os
import config
import numpy as np


def create_dirs(dirs):
    if 'data' not in os.listdir():
        os.system('mkdir data')
    for dir in dirs:
        if dir not in os.listdir('data'):
            os.system(f"mkdir {dir}")


def split_data(dataset_dir, val_size=0.2):
    images_path = os.path.join(dataset_dir, 'images')
    segmaps_path = os.path.join(dataset_dir, 'segmaps')
    file_names = np.array(os.listdir(images_path), dtype=object)

    dirs = [config.TRAIN_IMAGES_DIR, config.TRAIN_MAPS_DIR,
            config.VALID_IMAGES_DIR, config.VALID_MAPS_DIR]
    create_dirs(dirs)

    n = int(len(file_names) * val_size)
    valid_names = np.random.choice(file_names, n, replace=False)
    for name in file_names:
        img_fp = os.path.join(images_path, name)
        map_fp = os.path.join(segmaps_path, name)
        if name in valid_names:
            os.system(f"mv {img_fp} {config.VALID_IMAGES_DIR}")
            os.system(f"mv {map_fp} {config.VALID_MAPS_DIR}")
        else:
            os.system(f"mv {img_fp} {config.TRAIN_IMAGES_DIR}")
            os.system(f"mv {map_fp} {config.TRAIN_MAPS_DIR}")


if __name__ == "__main__":
    split_data('electron-microscopy-particle-segmentation', 0.2)
