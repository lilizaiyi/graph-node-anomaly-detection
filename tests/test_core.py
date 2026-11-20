import unittest
import numpy as np
from src.metrics import evaluate
from src.data import synthetic, validate


class CoreTests(unittest.TestCase):
    def test_metrics(self):
        y = np.array([0, 1, 0, 1])
        self.assertEqual(evaluate(y, y), {'roc_auc': 1., 'ap': 1.})
        self.assertEqual(evaluate(y, np.ones(4)), {'roc_auc': .5, 'ap': .5})
        self.assertEqual(evaluate(y, 1-y)['roc_auc'], 0.)
        mixed = evaluate(np.array([1, 0, 1]), np.array([.5, .5, .2]))
        self.assertAlmostEqual(mixed['roc_auc'], .25)
        self.assertAlmostEqual(mixed['ap'], 7/12)

    def test_data(self):
        x, a, y = synthetic()
        self.assertTrue(np.array_equal(a, a.T))
        self.assertEqual(y.sum(), 24)
        self.assertTrue(np.array_equal(x, synthetic()[0]))
        with self.assertRaises(ValueError):
            validate(x, a, np.ones(len(y)))


if __name__ == '__main__':
    unittest.main()
