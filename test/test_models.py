import unittest

import numpy as np

from models import create_model


class MyTestCase(unittest.TestCase):
    def test_resnet_18_creation(self):
        model = create_model("resnet18")
        self.assertEqual(model.input_shape, (16, 224, 224, 3))
        self.assertEqual(model.output_shape, (16, 512))

    def test_resnet_18_prediction_on_random(self):
        model = create_model("resnet18")
        prediction_random = model.predict(np.random.random((1, 224, 224, 3)))
        self.assertEqual(prediction_random.__class__, np.ndarray)
        self.assertFalse(np.all(prediction_random == np.zeros_like(prediction_random)))

    def test_resnet_18_prediction_on_zeros(self):
        model = create_model("resnet18")
        prediction_zeros = model.predict(np.zeros((1, 224, 224, 3)))
        self.assertEqual(prediction_zeros.__class__, np.ndarray)
        self.assertFalse(np.all(prediction_zeros == np.zeros_like(prediction_zeros)))


if __name__ == '__main__':
    unittest.main()
