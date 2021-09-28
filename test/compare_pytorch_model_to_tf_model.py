import unittest

import numpy as np
import torch

from models import create_model, create_pytorch_model


class MyTestCase(unittest.TestCase):

    def test_models_almost_equal(self):
        tf_model = create_model()
        pytorch_model = create_pytorch_model()
        random_inputs = np.random.random((1, 224, 224, 3))

        tf_outputs = tf_model.predict(random_inputs)

        pytorch_input_tensor = torch.from_numpy(random_inputs)
        pytorch_outputs = pytorch_model(pytorch_input_tensor)
        pytorch_outputs = pytorch_outputs.detach().numpy()
        np.testing.assert_array_almost_equal(tf_outputs, pytorch_outputs)


if __name__ == '__main__':
    unittest.main()
