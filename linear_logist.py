import numpy as np
from abstract_Lin_Model import LinearModel
from typing import Union

class LogisticRegression(LinearModel):
    """
    Класс логистической регрессии с регуляризацией, градиентным спуском и возможностью настройки вывода информации о процессе обучения.
    
    Attributes:
        learning_rate (float): Скорость обучения.
        max_iter (int): Максимальное количество итераций для обучения.
        weights (np.ndarray): Вектор весов модели.
        bias (float): Смещение модели.
        verbose (bool): Флаг для вывода информации о процессе обучения.
    """

    def fit(self, X: np.ndarray, y: Union[np.ndarray, list]) -> None:
        """
        Обучение модели логистической регрессии на основе входных данных.
        
        Args:
            X (np.ndarray): Матрица признаков формы (n_samples, n_features).
            y (Union[np.ndarray, list]): Вектор целевых значений формы (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.max_iter):
            y_pred = self._sigmoid(self._linear_output(X))
            error = y_pred - y
            gradient = (1 / n_samples) * (X.T @ error)
            gradient = self._apply_regularization(gradient)
            
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * (1 / n_samples) * np.sum(error)
            
            if self.verbose and i % 100 == 0:
                loss = self._log_loss(y, y_pred)
                print(f"Iteration {i}: Loss = {loss}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Выполняет предсказание классов для входной матрицы признаков.
        
        Args:
            X (np.ndarray): Матрица признаков формы (n_samples, n_features).
        
        Returns:
            np.ndarray: Вектор предсказанных классов (0 или 1) формы (n_samples,).
        """
        linear_output = self._linear_output(X)
        return (self._sigmoid(linear_output) >= 0.5).astype(int)

    def _linear_output(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисляет линейное представление выходных данных для заданной матрицы признаков.
        
        Args:
            X (np.ndarray): Матрица признаков формы (n_samples, n_features).
        
        Returns:
            np.ndarray: Вектор линейного представления формы (n_samples,).
        """
        return X @ self.weights + self.bias

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Применяет сигмоидную функцию к входному значению.
        
        Args:
            z (np.ndarray): Входной массив.
        
        Returns:
            np.ndarray: Массив значений, обработанных сигмоидной функцией.
        """
        return 1 / (1 + np.exp(-z))
    
    def _log_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Вычисляет логистическую функцию потерь (log loss) с учётом регуляризации.
        
        Args:
            y (np.ndarray): Вектор истинных меток формы (n_samples,).
            y_pred (np.ndarray): Вектор предсказанных вероятностей формы (n_samples,).
        
        Returns:
            float: Значение логистической функции потерь с учётом регуляризации.
        """
        epsilon = 1e-15  
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) + self._l1_penalty() + self._l2_penalty()
