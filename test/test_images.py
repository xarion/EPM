import unittest

import numpy as np

from dataset import get_validation_dataset


class MyTestCase(unittest.TestCase):
    def test_data_is_normalized(self):
        ds = get_validation_dataset(batch_size=1)
        data = ds.__iter__().next().numpy()
        self.assertLessEqual(np.max(data), 255)
        # 90 is a magic number here, i want to ensure that the images are not scaled to [-1,1] or [0,1]
        self.assertLess(90, np.max(data))
        self.assertLessEqual(0, np.min(data))


if __name__ == '__main__':
    unittest.main()
