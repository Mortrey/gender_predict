import pandas as pd

from collections import Counter
import pickle
import os


unique_keys = {'orders', 'visits', 'last-visits-in-categories', 'exchange-sessions', 'site-meta'}

folder_path = "./grid_dict"

print('Загружаем жадные словари...')

# Получаем список файлов в папке
files = os.listdir(folder_path)

for file in files:
    # Проверяем, что файл имеет расширение .pickle
    if file.endswith(".pickle"):
        # Получаем имя файла без расширения
        file_name = os.path.splitext(file)[0]

        # Формируем полный путь к файлу
        file_path = os.path.join(folder_path, file)

        # Загружаем объект из файла
        with open(file_path, "rb") as pickle_file:
            # Используем имя файла без расширения в качестве переменной для загруженного объекта
            locals()[file_name] = pickle.load(pickle_file)

# Создание столбца predict_gender учитывет значимость элемента
def predict_gender(row):
    male_score = 0
    female_score = 0
    significant_multiplier = 2.25  # Усиление значимого скора
    penalty_multiplier = 0.4    # Штраф незначимого скора
    coef = 1

    # Обработка orders_count
    for site_id, count in row['orders_count'].items():
        if site_id in significant_orders_count:
            coef = significant_multiplier 
        else:
            coef = penalty_multiplier
        if site_id in gender_ratio_orders_count:
            male_score += gender_ratio_orders_count[site_id]['male'] * count * coef 
            female_score += gender_ratio_orders_count[site_id]['female'] * count * coef

    # Обработка visits_count
    for site_id, count in row['visits_count'].items():
        if site_id in significant_visits_count:
            coef = significant_multiplier 
        else:
            coef = penalty_multiplier
        if site_id in gender_ratio_visits_count:
            male_score += gender_ratio_visits_count[site_id]['male'] * count * coef
            female_score += gender_ratio_visits_count[site_id]['female'] * count * coef

    # Обработка last_visited_categories - лучше убрать!
    #for category in row['last_visited_categories']:
    #    if category in significant_last_visited_categories:
    #        coef = significant_multiplier
    #    else:
    #        coef = penalty_multiplier
    #    if category in gender_ratio_last_visited_categories:
    #        male_score += gender_ratio_last_visited_categories[category]['male'] * coef 
    #        female_score += gender_ratio_last_visited_categories[category]['female'] * coef

    # Обработка site_meta_list
    for site_id in row['site_meta_list']:
        if site_id in significant_site_meta_list:
            coef = significant_multiplier * 3 # регистрация на сайте важный признак его нужно максить
        else:
            coef = penalty_multiplier * 8 # и не штрафовать даже если он не в списке
        if site_id in gender_ratio_site_meta_list:
            male_score += gender_ratio_site_meta_list[site_id]['male'] * coef
            female_score += gender_ratio_site_meta_list[site_id]['female'] * coef

    # Обработка item_ids_count
    for item_id, count in row['item_ids_count'].items():
        if site_id in significant_item_ids_count:
            coef = significant_multiplier
        else:
            coef = penalty_multiplier
        if item_id in gender_ratio_item_ids_count:
            male_score += gender_ratio_item_ids_count[item_id]['male'] * count * coef
            female_score += gender_ratio_item_ids_count[item_id]['female'] * count * coef

    # Обработка category_path_count - это тоже лучше убрать
    #for category_id, count in row['category_path_count'].items():
    #    if category_id in significant_category_path_count:
    #        coef = significant_multiplier
    #    else:
    #        coef = penalty_multiplier
    #    if category_id in gender_ratio_category_path_count:
    #        male_score += gender_ratio_category_path_count[category_id]['male'] * count * coef
    #        female_score += gender_ratio_category_path_count[category_id]['female'] * count * coef

    # Обработка brand_ids_count
    for brand_id, count in row['brand_ids_count'].items():
        if site_id in significant_brand_ids_count:
            coef = significant_multiplier
        else:
            coef = penalty_multiplier
        if brand_id in gender_ratio_brand_ids_count:
            male_score += gender_ratio_brand_ids_count[brand_id]['male'] * count * coef
            female_score += gender_ratio_brand_ids_count[brand_id]['female'] * count * coef

    # Обработка selected_sites
    #for site_id in row['selected_sites']:
    #    if site_id in significant_selected_sites:
    #        coef = significant_multiplier
    #    else:
    #        coef = penalty_multiplier
    #    if site_id in gender_ratio_selected_sites:
    #        male_score += gender_ratio_selected_sites[site_id]['male'] * coef
    #        female_score += gender_ratio_selected_sites[site_id]['female'] * coef

    # мужчины мало бегают по интернету поможем коэффициентом
    return 'female' if female_score > male_score * 1.49 else 'male' 



# Функция рассчитывает определенные айди сайтов в колонке заказы посчитает сколько раз наш пользователь заказывал что то на этом сайте
def count_site_ids_in_orders_modified(orders):
    site_id_counter = Counter()
    if orders is not None:
        for order_dict in orders:
            if 'orders' in order_dict and 'site-id' in order_dict:
                site_id = order_dict['site-id']
                site_id_counter[site_id] = len(order_dict['orders'])
    return site_id_counter


# теперь тоже самое сделаем и для колонки с посещениями сайтов - посмотрим куда человек ходит и как часто
def count_site_ids_in_visits_modified(visits):
    site_id_counter = Counter()
    if visits is not None:
        for visit_group in visits:
            if 'visits' in visit_group and 'site-id' in visit_group:
                site_id = visit_group['site-id']
                site_id_counter[site_id] = len(visit_group['visits'])
    return site_id_counter


# теперь для каждого пользвателя мы хотим вытащить последние категории посещений
def extract_categories(last_visits):
    if last_visits is not None:
        return [visit['category'] for visit in last_visits]
    return []


# Функция для извлечения списка site-id из колонки site-meta
def extract_site_meta(site_meta):
    if site_meta is not None:
        return [meta['site-id'] for meta in site_meta]
    return []


# эта функция посчитает количество предметов купленных пользователем и сгруппирует их по их номеру
def count_item_ids_in_orders(orders):
    item_id_counter = Counter()
    if orders is not None:
        for order_dict in orders:
            for order in order_dict.get('orders', []):
                for item in order.get('items', []):
                    item_id = item.get('id')
                    count = item.get('count', 1)  # Устанавливаем значение по умолчанию 1
                    if count is not None:  # Дополнительная проверка на None
                        item_id_counter[item_id] += count
    return item_id_counter


# эта функция в том же накопительном режиме спарсит категории товаров, которые покупал пользователь
def count_categories_in_orders(orders):
    category_counter = Counter()
    if orders is not None:
        for order_dict in orders:
            for order in order_dict.get('orders', []):
                for item in order.get('items', []):
                    categories = item.get('general-category-path', [])
                    for category in categories:
                        category_counter[category] += 1
    return category_counter


# Эта функция займется подсчетом бредов, тоже показтельно
def count_brand_ids_in_orders(orders):
    brand_id_counter = Counter()
    if orders is not None:
        for order_dict in orders:
            for order in order_dict.get('orders', []):
                for item in order.get('items', []):
                    brand_id = item.get('brand-id')
                    count = item.get('count', 1)  # Устанавливаем значение по умолчанию 1
                    if brand_id is not None and count is not None:  # Дополнительная проверка на None
                        brand_id_counter[brand_id] += count
    return brand_id_counter


# Функция для извлечения списка выбранных сайтов из колонки exchange-sessions
def extract_selected_sites(exchange_sessions):
    selected_sites = []
    if exchange_sessions is not None:
        for session in exchange_sessions:
            accepted_site_id = session.get('accepted-site-id')
            if accepted_site_id is not None:
                selected_sites.append(accepted_site_id)
    return selected_sites


# ну вот тут не грех сделать и функцию, тем более что она пригодится на "продакшен файле"
def test_data(df):
    df = df.T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'user_id', 0: 'features'}, inplace=True)
    for key in unique_keys:
        df[key] = df['features'].apply(lambda x: x.get(key))
    df.drop('features', axis=1, inplace=True)
    df['orders_count'] = df['orders'].apply(count_site_ids_in_orders_modified)
    df['visits_count'] = df['visits'].apply(count_site_ids_in_visits_modified)
    df['last_visited_categories'] = df['last-visits-in-categories'].apply(extract_categories)
    df['site_meta_list'] = df['site-meta'].apply(extract_site_meta)
    df['item_ids_count'] = df['orders'].apply(count_item_ids_in_orders)
    df['category_path_count'] = df['orders'].apply(count_categories_in_orders)
    df['brand_ids_count'] = df['orders'].apply(count_brand_ids_in_orders)
    df['selected_sites'] = df['exchange-sessions'].apply(extract_selected_sites)
    df = df.drop(['visits', 'orders', 'last-visits-in-categories',
       'exchange-sessions', 'site-meta'], axis=1)
    return df