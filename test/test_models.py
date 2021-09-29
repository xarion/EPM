import os
import unittest

import PIL
import numpy as np
import torch

from models import create_feature_extractor_model, create_pytorch_feature_extractor


class MyTestCase(unittest.TestCase):
    def test_resnet_18_creation(self):
        model = create_feature_extractor_model()
        self.assertEqual(model.input_shape, (16, 224, 224, 3))
        self.assertEqual(model.output_shape, (16, 2048))

    def test_resnet_18_prediction_on_random(self):
        model = create_feature_extractor_model()
        prediction_random = model.predict(np.random.random((1, 224, 224, 3)))
        self.assertEqual(prediction_random.__class__, np.ndarray)
        self.assertFalse(np.all(prediction_random == np.zeros_like(prediction_random)))

    def test_resnet_18_prediction_on_zeros(self):
        model = create_feature_extractor_model()
        prediction_zeros = model.predict(np.zeros((1, 224, 224, 3)))
        self.assertEqual(prediction_zeros.__class__, np.ndarray)
        self.assertFalse(np.all(prediction_zeros == np.zeros_like(prediction_zeros)))

    def test_pytorch_image_saver(self):
        model = create_pytorch_feature_extractor("test_model")
        image = np.random.randint(low=0, high=255, size=(1, 223, 224, 3)).astype(float)

        t_image = torch.from_numpy(image.copy())
        out = model(t_image)
        self.assertTrue(os.path.exists('test_model.1.png'))
        read_image = PIL.Image.open("test_model.1.png")
        read_image_data = np.array(read_image)
        np.testing.assert_array_equal(read_image_data, image[0])


if __name__ == '__main__':
    unittest.main()
