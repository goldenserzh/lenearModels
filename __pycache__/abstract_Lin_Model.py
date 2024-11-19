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

    def update_weights(self):
    """
    Обновляет веса (W) и смещение (b) логистической регрессии с использованием 
    градиентного спуска и опциональной L1- и L2-регуляризации.

    Градиенты рассчитываются на основе функции логистической потери:
    - Потеря = Кросс-энтропийная потеря + Потеря L1-регуляризации + Потеря L2-регуляризации

    Градиенты для весов и смещения:
    - dW = Градиент функции потерь по весам W
    - db = Градиент функции потерь по смещению b

    Обновление параметров происходит следующим образом:
    - W = W - learning_rate * dW
    - b = b - learning_rate * db

    Регуляризация:
    - L1-регуляризация добавляет штраф за сумму абсолютных значений весов (|W|).
    - L2-регуляризация добавляет штраф за сумму квадратов весов (W^2).

    Возвращаемое значение:
    - Метод изменяет параметры модели на месте (W и b) и ничего не возвращает.
    """
    pass
