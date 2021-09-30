import unittest

import numpy as np
from torch.utils.data import DataLoader

from config import IMAGE_CLASS
from dataset import get_validation_dataset
from models import create_model


class MyTestCase(unittest.TestCase):
    def test_something(self):
        ds = get_validation_dataset()
        dl = DataLoader(ds, batch_size=50)
        model, _ = create_model()
        torch_predictions = model(next(iter(dl))[0].cuda())
        predictions = torch_predictions.detach().cpu().numpy()
        class_predictions_for_all = np.sum(predictions, axis=0)
        max_predicted_class = np.argmax(class_predictions_for_all)
        self.assertEqual(max_predicted_class, IMAGE_CLASS)


if __name__ == '__main__':
    unittest.main()
