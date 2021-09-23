import tensorflow as tf

from classification_models.tfkeras import Classifiers


def create_model(model_name):
    classifier, preprocess_input = Classifiers.get(model_name)
    model = classifier((224, 224, 3), weights='imagenet', include_top=False)

    input = tf.keras.Input((224, 224, 3), batch_size=16)

    output = model(input)
    output = tf.keras.layers.GlobalAveragePooling2D()(output)

    return tf.keras.models.Model(input, output)


# test small change
