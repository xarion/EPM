from os.path import join

import tensorflow as tf
from classification_models.tfkeras import Classifiers

from config import MODEL_NAME


def create_feature_extractor_model():
    classifier, preprocess_input = Classifiers.get(MODEL_NAME)
    model = classifier((224, 224, 3), weights='imagenet', include_top=False)

    image_input = tf.keras.Input((224, 224, 3), batch_size=16, name="image")
    image_input = preprocess_input(image_input)
    model_output = model(image_input)
    model_output = tf.keras.layers.GlobalAveragePooling2D()(model_output)

    return tf.keras.models.Model(image_input, model_output)


def create_adjusted_image_saving_model_for_xai(xai_name):
    classifier, preprocess_input = Classifiers.get(MODEL_NAME)
    model = classifier((224, 224, 3), weights='imagenet', include_top=True)

    first_image_input = tf.keras.Input((224, 224, 3), batch_size=1, name="image")
    image_input = SaveLayer(xai_name)(first_image_input)
    image_input = preprocess_input(image_input)
    model_output = model(image_input)

    return tf.keras.models.Model(first_image_input, model_output)


class SaveLayer(tf.keras.layers.Layer):

    def __init__(self, xai_model_name, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.xai_model_name = xai_model_name
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def build(self, input_shape):
        # tf.io.gfile.mkdir(self.xai_model_name)
        self.counter = tf.Variable(0, dtype="int32")

    def call(self, inputs, *args, **kwargs):
        batch_image_input = inputs

        def __save_each_image(image_input):
            self.counter.count_up_to(tf.int32.max)
            file_path = tf.strings.join(
                [self.xai_model_name, tf.strings.as_string(self.counter), "jpg"], separator='.', name=None
            )
            int8_image = tf.cast(image_input, tf.uint8)
            jpeg_encoded_input = tf.io.encode_jpeg(int8_image)
            write_op = tf.io.write_file(file_path, jpeg_encoded_input)
            with tf.control_dependencies([write_op]):
                return image_input
        saved_image_inputs = tf.map_fn(__save_each_image, batch_image_input)
        return batch_image_input
