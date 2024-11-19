README
Linear Models Project
Проект предоставляет реализацию линейных моделей машинного обучения для решения задач регрессии и классификации. Основой проекта является абстрактный класс LinearModel, от которого наследуются классы LinearRegression и LogitRegression. Эти классы содержат методы для обучения, предсказания и оценки моделей, а также поддерживают регуляризацию.
Классы
1. LinearModel (Абстрактный класс)
LinearModel задает структуру для реализации линейных моделей. Этот класс включает общие параметры модели и определяет базовые методы, которые необходимо реализовать в дочерних классах.
•	Атрибуты:
o	learning_rate (float): Скорость обучения.
o	num_of_itr (int): Количество итераций обучения.
•	Абстрактные методы:
o	fit(x, y): Обучение модели на данных.
o	predict(x): Предсказание на основе обученной модели.
o	loss(y_true, y_pred): Вычисление ошибки модели.
2. LinearRegression (Дочерний класс)
LinearRegression — это реализация линейной регрессии с возможностью использования L1- и L2-регуляризации. Подходит для задач регрессии и может управлять выводом информации о процессе обучения.
•	Атрибуты:
o	learning_rate (float): Скорость обучения.
o	num_of_itr (int): Количество итераций обучения.
o	l1 (float): Коэффициент L1-регуляризации.
o	l2 (float): Коэффициент L2-регуляризации.
o	verbose (bool): Флаг для вывода информации об обучении.
•	Методы:
o	fit(x, y): Обучает модель на данных, используя метод градиентного спуска с учетом регуляризации.
o	predict(x): Выполняет предсказание на новых данных.
o	MSE(x, y): Рассчитывает среднеквадратичную ошибку (MSE) с учетом регуляризации.
o	update_weights(): Обновляет веса модели.
o	train_test_split(x, y, test_size): Разделяет данные на обучающий и тестовый наборы.
o	analytical_solution(x, y): Рассчитывает веса аналитическим методом с использованием L2-регуляризации.
o	loss(x, y): Вычисляет функцию потерь, использующую MSE.
3. LogitRegression (Дочерний класс)
LogitRegression реализует логистическую регрессию для задач классификации. Поддерживает L1- и L2-регуляризацию и контроль за выводом процесса обучения.
•	Атрибуты:
o	learning_rate (float): Скорость обучения.
o	iterations (int): Количество итераций обучения.
o	l1 (float): Коэффициент L1-регуляризации.
o	l2 (float): Коэффициент L2-регуляризации.
o	verbose (bool): Флаг для вывода информации об обучении.
•	Методы:
o	fit(X, Y): Обучает модель на данных, используя метод градиентного спуска с регуляризацией.
o	update_weights(): Обновляет веса модели.
o	loss(): Вычисляет функцию потерь с учетом L1- и L2-регуляризации.
o	predict(X): Выполняет предсказание (выдаёт метки классов).
o	train_test_split(x, y, test_size): Разделяет данные на обучающий и тестовый наборы.
Вспомогательные функции
•	normalize_features(X): Нормализует признаки, чтобы сделать их совместимыми для обучения моделей.
Установка и зависимости
Для установки зависимостей:
bash
Copy code
pip install -r requirements.txt


