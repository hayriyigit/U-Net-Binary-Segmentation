import os
import config
from dataset import Dataset, DataLoader
import tensorflow as tf
import segmentation_models as sm


def get_model():
    model = sm.Unet(config.BACKBONE,
                    classes=config.NUM_CLASSES,
                    activation=config.ACTIVATION)

    optim = tf.keras.optimizers.Adam(config.LR)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    metrics = [sm.metrics.IOUScore(
        threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model.compile(optim, total_loss, metrics)

    return model


def main():
    # LOAD DATA
    preprocess_input = sm.get_preprocessing(config.BACKBONE)

    # Dataset for train images
    train_dataset = Dataset(
        config.TRAIN_IMAGES_DIR,
        config.TRAIN_MAPS_DIR,
        augmentation=config.transforms,
        preprocessing=config.get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        config.VALID_IMAGES_DIR,
        config.TRAIN_MAPS_DIR,
        augmentation=config.transforms,
        preprocessing=config.get_preprocessing(preprocess_input),
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # LOAD MODEL
    model = get_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.WEIGHT_FILE, save_weights_only=True, save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau()
    ]

    if config.LOAD_WEIGHTS and config.WEIGHT_FILE in os.listdir():
        print("=> Loading checkpoint ...")
        model.load_weights('best_model.h5')

    model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=config.EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )


if __name__ == "__main__":
    sm.set_framework('tf.keras')
    main()
