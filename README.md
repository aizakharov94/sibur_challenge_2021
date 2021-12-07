## Sibur challenge 2021 (Предсказание спроса)

Top 1 private solution (1.4027 public and 1.4073 private) <br/>
Link: [Sibur challenge 2021](https://sibur.ai-community.com/competitions/5/tasks/13/rating)

#### Структура репозитория:

1) Fit_pipeline.ipynb - обучение модели и получение обученной модели для инференса. <br/>
&nbsp;&nbsp;&nbsp;&nbsp; На выходе получаем: <br/>
|-- ohe.pkl - One Hot Encoder <br/>
|-- model.pkl - Модель регрессии (LGBM) <br/>
|-- model_class_0.pkl - Модель классификации (LGBM) <br/>

2) Inference.ipynb - применение файла predict.py для прогноза спроса на следующий месяц. <br/>
3) predict.py - файл с инференсом, который принимает модели с Fit_pipeline.ipynb и применения для предсказания спроса на следующий месяц. <br/>
4) requirements.txt - версии необходимых библиотек <br/>
