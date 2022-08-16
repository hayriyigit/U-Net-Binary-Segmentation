import cv2
import Albumentations as A

VALID_SIZE = 0.2
BACKBONE = 'efficientnetb3'
ACTIVATION = 'sigmoid'
WEIGHT_FILE = 'best_model.h5'
BATCH_SIZE = 8
NUM_CLASSES = 1
LR = 1e-4
EPOCHS = 50
WORKERS = 4
TRAIN_IMAGES_DIR = 'data/train_images'
TRAIN_MAPS_DIR = 'data/train_maps'
VALID_IMAGES_DIR = 'data/valid_images'
VALID_MAPS_DIR = 'data/valid_maps'
LOAD_WEIGHTS = False

transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
    A.RandomBrightnessContrast(
        contrast_limit=0.3, brightness_limit=0.3, p=0.2),
    A.OneOf([
        A.ImageCompression(p=0.8),
        A.RandomGamma(p=0.8),
        A.Blur(p=0.8),
    ], p=1.0),
    A.OneOf([
        A.ImageCompression(p=0.8),
        A.RandomGamma(p=0.8),
        A.Blur(p=0.8),
    ], p=1.0),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
])


def get_preprocessing(preprocessing_fn):
    return A.Compose([
        A.Lambda(image=preprocessing_fn),
    ])