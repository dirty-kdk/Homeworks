import unittest
import numpy as np
from homework_experiments import create_polynomial_features, create_interaction_features, create_statistical_features

class TestFeatureEngineering(unittest.TestCase):
    def test_polynomial_features(self):
        """Тест создания полиномиальных признаков."""
        X = np.array([[1, 2], [3, 4]])
        X_poly = create_polynomial_features(X, degree=2)
        self.assertEqual(X_poly.shape[1], 5)  # 2 признака + их квадраты + взаимодействие
        self.assertAlmostEqual(X_poly[0, 0], 1.0)  # Проверка значения

    def test_interaction_features(self):
        """Тест создания признаков взаимодействий."""
        X = np.array([[1, 2], [3, 4]])
        X_inter = create_interaction_features(X)
        self.assertEqual(X_inter.shape[1], 1)  # Одно взаимодействие для двух признаков
        self.assertAlmostEqual(X_inter[0, 0], 2.0)  # 1 * 2

    def test_statistical_features(self):
        """Тест создания статистических признаков."""
        X = np.array([[1, 2], [3, 4]])
        X_stat = create_statistical_features(X)
        self.assertEqual(X_stat.shape[1], 2)  # Среднее и дисперсия
        self.assertAlmostEqual(X_stat[0, 0], 1.5)  # Среднее (1 + 2) / 2

if __name__ == '__main__':
    unittest.main()