import pandas as pd
import numpy as np
import abc
from typing import Union

class LinearModel(abc.ABC):
    """
    Абстрактный базовый класс для создания линейных моделей.

    Attributes:
        learning_rate (float): Скорость обучения модели.
        num_of_itr (int): Количество итераций для процесса обучения.
    """

    def __init__(self, learning_rate: float, num_of_itr: int) -> None:
        """
        Инициализирует параметры обучения.

        Args:
            learning_rate (float): Скорость обучения.
            num_of_itr (int): Количество итераций.
        """
        self.learning_rate = learning_rate
        self.num_of_itr = num_of_itr

    @abc.abstractmethod
    def fit(self, x: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Абстрактный метод для подгонки модели под данные.
        
        Args:
            x (Union[np.ndarray, pd.DataFrame]): Массив признаков.
            y (Union[np.ndarray, pd.Series]): Целевые значения.
        
        Raises:
            NotImplementedError: Этот метод должен быть реализован в дочернем классе.
        """
        pass

    @abc.abstractmethod
    def predict(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Абстрактный метод для выполнения предсказания на основе обученной модели.
        
        Args:
            x (Union[np.ndarray, pd.DataFrame]): Массив признаков для предсказания.
        
        Returns:
            np.ndarray: Предсказанные значения.
        
        Raises:
            NotImplementedError: Этот метод должен быть реализован в дочернем классе.
        """
        pass

    @abc.abstractmethod
    def loss(self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray) -> float:
        """
        Абстрактный метод для вычисления ошибки модели.
        
        Args:
            y_true (Union[np.ndarray, pd.Series]): Истинные значения целевой переменной.
            y_pred (np.ndarray): Предсказанные значения модели.
        
        Returns:
            float: Значение ошибки.
        
        Raises:
            NotImplementedError: Этот метод должен быть реализован в дочернем классе.
        """
        pass
