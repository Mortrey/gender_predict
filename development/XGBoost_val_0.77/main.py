import json
import sys
import joblib
import pandas as pd
from process_func import *


# Чтение категорий из файлов CSV
categories_df = pd.read_csv('cats/categories.csv')
last_categories_df = pd.read_csv('cats/last_categories.csv')

# Преобразование DataFrame обратно в список
all_categories = categories_df['Category'].tolist()
unique_last_cats = last_categories_df['Category'].tolist()


# Функция для загрузки модели
def load_model(model_path):
    return joblib.load(model_path)


# Функция для обработки входных данных и выполнения предсказания
def predict(input_json, model):
    # Преобразования JSON
    merge_df_test = get_df(input_json, target=False)
    X_test = preprocessing_df(merge_df_test, all_categories, unique_last_cats, target=False)

    # Выполнение предсказания
    prediction = model.predict(X_test)
    y_pred_label = ['female' if pred == 1 else 'male' for pred in prediction][0]

    # Возвращаем предсказание в виде строки
    return y_pred_label


if __name__ == "__main__":
    # Путь к файлу модели
    model_path = 'model/trained_pipeline.pkl'

    # Загрузка модели
    model = load_model(model_path)

    # Чтение JSON из файла
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        with open(json_path, 'r') as file:
            input_json = json.load(file)
    else:
        print("Напишите путь до файла json")
        sys.exit(1)

    result = predict(input_json, model)
    print(result)
