## Sibur challenge 2021 (Предсказание спроса)

Top 1 private solution (1.4027 public and 1.4073 private)
Link: ###### [Sibur challenge 2021](https://sibur.ai-community.com/competitions/5/tasks/13/rating)

#### Структура репозитория:

1) Fit_pipeline.ipynb - обучение модели и получение обученной модели для инференса.

&nbsp;&nbsp;&nbsp;&nbsp; На выходе получаем:
|-- ohe.pkl - One Hot Encoder
|-- model.pkl - Модель регрессии (LGBM)
|-- model_class_0.pkl - Модель классификации (LGBM)

2) Inference.ipynb - применение файла predict.py для прогноза спроса на следующий месяц.

3) predict.py - файл с инференсом, который принимает модели с Fit_pipeline.ipynb и применения для предсказания спроса на следующий месяц.

4) requirements.txt - версии необходимых библиотек
