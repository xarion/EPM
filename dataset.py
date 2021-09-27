import tensorflow as tf
import tensorflow_datasets as tfds

from config import IMAGE_CLASS


def get_validation_dataset(batch_size=16):
    ds = tfds.load("imagenet2012", split="validation", shuffle_files=False)
    ds = ds.filter(lambda row: row["label"] == IMAGE_CLASS)

    ds = ds.map(lambda row: tf.image.resize(row["image"], (224, 224)))
    if batch_size is not None:
        ds = ds.batch(batch_size)
    return ds

def get_train_dataset(batch_size=16):
    ds = tfds.load("imagenet2012", split="train", shuffle_files=False, download=True)
    ds = ds.filter(lambda row: row["label"] == IMAGE_CLASS)

    ds = ds.map(lambda row: tf.image.resize(row["image"], (224, 224)))
    if batch_size is not None:
        ds = ds.batch(batch_size)
    return ds
