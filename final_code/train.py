# блок импортов
print('Импортируем необходимые библиотеки')
import pandas as pd
from collections import Counter
import pickle
import timeit

start_time = timeit.default_timer()

start_time_block1 = timeit.default_timer()

# зададим директорию откуда будет забирать тренировочный набор данных и загрузим их
print('Загружаем тренировочный набор данных и производим над ним преобразования ......')
file_path_train = "C:\\Users\\HP\\Desktop\\hacaton\\data\\train.json"

# загружаем наш тренировочный набор данных в pandas
df_train = pd.read_json(file_path_train)

# транспонируем датасет что бы он принял человеческий вид
df_train = df_train.T

# зададим имена колонок
df_train.reset_index(inplace=True)
df_train.rename(columns={'index': 'user_id', 0: 'target', 1: 'features'}, inplace=True)

# разобьем данные по уникальным признакам в колонке features
unique_keys = set()
for features in df_train['features']:
    unique_keys.update(features.keys()) # просто вытащим их в форме множества

# создадим из них новые колонки извлекая данные из 'features'
for key in unique_keys:
    df_train[key] = df_train['features'].apply(lambda x: x.get(key)) # да просто обратимся по ключу
# колонку 'features' удалим за дальнейшей ненадобностью
df_train.drop('features', axis=1, inplace=True)

end_time_block1 = timeit.default_timer()
execution_time_block1 = round(end_time_block1 - start_time_block1, 2)
print(f"Время выполнения блока кода 1: {execution_time_block1} секунд")

start_time_block2 = timeit.default_timer()

print('Активируем функции для парсинга данных и выполняем парсинг .....')

# Функция рассчитывает определенные айди сайтов в колонке заказы посчитает сколько раз наш пользователь заказывал что то на этом сайте
def count_site_ids_in_orders_modified(orders):
    site_id_counter = Counter()
    if orders is not None:
        for order_dict in orders:
            if 'orders' in order_dict and 'site-id' in order_dict:
                site_id = order_dict['site-id']
                for order in order_dict['orders']:
                    site_id_counter[site_id] += 1
    return site_id_counter

# теперь тоже самое сделаем и для колонки с посещениями сайтов - посмотрим куда человек ходит и как часто
def count_site_ids_in_visits_modified(visits):
    site_id_counter = Counter()
    if visits is not None:
        for visit_group in visits:
            if 'visits' in visit_group and 'site-id' in visit_group:
                site_id = visit_group['site-id']
                for visit in visit_group['visits']:
                    site_id_counter[site_id] += 1
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

# Создадим новую колонку в которой будет количество заказов на каждом сайте для каждого пользователя
df_train['orders_count'] = df_train['orders'].apply(count_site_ids_in_orders_modified)

# Применим функцию и создадим новый столбец с посещениями сайтов
df_train['visits_count'] = df_train['visits'].apply(count_site_ids_in_visits_modified)

# применим функцию и создадим новый столбец с требуемым списком
df_train['last_visited_categories'] = df_train['last-visits-in-categories'].apply(extract_categories)

# Применение функции к столбцу site-meta
df_train['site_meta_list'] = df_train['site-meta'].apply(extract_site_meta)

# Применение функции к заказам - вытащим товры
df_train['item_ids_count'] = df_train['orders'].apply(count_item_ids_in_orders)

# тут выащим категорию товара
df_train['category_path_count'] = df_train['orders'].apply(count_categories_in_orders)

# ну и бренды, на самом деле они мне кажется мало информативны, но все таки посмотрим
df_train['brand_ids_count'] = df_train['orders'].apply(count_brand_ids_in_orders)

# Применение функции к DataFrame, вытащим что же наши пользователи выбирали
df_train['selected_sites'] = df_train['exchange-sessions'].apply(extract_selected_sites)

# удалим колонки исходники за ненадобнотстью:)
df_train = df_train.drop(['visits', 'orders', 'last-visits-in-categories',
       'exchange-sessions', 'site-meta'], axis=1)

print('Данные преобразованы необходимым образом')

end_time_block2 = timeit.default_timer()
execution_time_block2 = round(end_time_block1 - start_time_block2, 2)
print(f"Время выполнения блока кода 2: {execution_time_block2} секунд")

start_time_block3 = timeit.default_timer()
print('Преступаем к созданию жадных словарей .....')

# Инициализация коэффициентов значимости для каждого столбца - тут хорошо бы грид серч но мощностей нет,
# работаем с единичками
coefficients = {
    'orders_count_coeff': 1,
    'visits_count_coeff': 1,
    'last_visited_categories_coeff': 1,
    'site_meta_list_coeff': 1,
    'item_ids_count_coeff': 1,
    'category_path_count_coeff': 1,
    'brand_ids_count_coeff': 1,
    'selected_sites_coeff': 1
}

# Инициализация статистических словарей 
orders_count_stats = {}
visits_count_stats = {}
last_visited_categories_stats = {}
site_meta_list_stats = {}
item_ids_count_stats = {}
category_path_count_stats = {}
brand_ids_count_stats = {}
selected_sites_stats = {}

# Проход по каждой строке DataFrame и мы заполним наши словари в логике ['ключ']:{male:0, female:0}

for _, row in df_train.iterrows():
    gender = row['target']
    
    # Обработка orders_count
    for site_id, count in row['orders_count'].items():
        if site_id not in orders_count_stats:
            orders_count_stats[site_id] = {'male': 0, 'female': 0}
        orders_count_stats[site_id][gender] += count * coefficients['orders_count_coeff']

    # Обработка visits_count
    for site_id, count in row['visits_count'].items():
        if site_id not in visits_count_stats:
            visits_count_stats[site_id] = {'male': 0, 'female': 0}
        visits_count_stats[site_id][gender] += count * coefficients['visits_count_coeff']

    # Обработка last_visited_categories
    for category in row['last_visited_categories']:
        if category not in last_visited_categories_stats:
            last_visited_categories_stats[category] = {'male': 0, 'female': 0}
        last_visited_categories_stats[category][gender] += 1 * coefficients['last_visited_categories_coeff']

    # Обработка site_meta_list
    for site_id in row['site_meta_list']:
        if site_id not in site_meta_list_stats:
            site_meta_list_stats[site_id] = {'male': 0, 'female': 0}
        site_meta_list_stats[site_id][gender] += 1 * coefficients['site_meta_list_coeff']

    # Обработка item_ids_count
    for item_id, count in row['item_ids_count'].items():
        if item_id not in item_ids_count_stats:
            item_ids_count_stats[item_id] = {'male': 0, 'female': 0}
        item_ids_count_stats[item_id][gender] += count * coefficients['item_ids_count_coeff']

    # Обработка category_path_count
    for category_id, count in row['category_path_count'].items():
        if category_id not in category_path_count_stats:
            category_path_count_stats[category_id] = {'male': 0, 'female': 0}
        category_path_count_stats[category_id][gender] += count * coefficients['category_path_count_coeff']

    # Обработка brand_ids_count
    for brand_id, count in row['brand_ids_count'].items():
        if brand_id not in brand_ids_count_stats:
            brand_ids_count_stats[brand_id] = {'male': 0, 'female': 0}
        brand_ids_count_stats[brand_id][gender] += count * coefficients['brand_ids_count_coeff']

    # Обработка selected_sites
    for site_id in row['selected_sites']:
        if site_id not in selected_sites_stats:
            selected_sites_stats[site_id] = {'male': 0, 'female': 0}
        selected_sites_stats[site_id][gender] += 1 * coefficients['selected_sites_coeff']

# Заполнение словарей для расчета доли мужского и женского участия через фунцию
def calculate_gender_ratio(stats_dict):
    gender_ratio_dict = {}
    for key, stats in stats_dict.items():
        total = stats['male'] + stats['female']
        if total > 0:
            gender_ratio_dict[key] = {
                'male': round(stats['male'] / total, 10),
                'female': round(stats['female'] / total, 10)
            }
    return gender_ratio_dict

# Просто применим функцию и заполним наши словари
gender_ratio_orders_count = calculate_gender_ratio(orders_count_stats)
gender_ratio_visits_count = calculate_gender_ratio(visits_count_stats)
gender_ratio_last_visited_categories = calculate_gender_ratio(last_visited_categories_stats)
gender_ratio_site_meta_list = calculate_gender_ratio(site_meta_list_stats)
gender_ratio_item_ids_count = calculate_gender_ratio(item_ids_count_stats)
gender_ratio_category_path_count = calculate_gender_ratio(category_path_count_stats)
gender_ratio_brand_ids_count = calculate_gender_ratio(brand_ids_count_stats)
gender_ratio_selected_sites = calculate_gender_ratio(selected_sites_stats)

print('Записываем наши словари')
# создадим функцию для записи словаря в формат pickle
def dict_to_pickle(dict_name, file_name):
    # Сохранение словаря в файл с использованием pickle
    with open(file_name + '.pickle', 'wb') as file:
        pickle.dump(dict_name, file)

# сохраним эти словари - это мотор для нашей модели классификации - поэтому нужно запомнить куда мы их сохраняем
dict_to_pickle(gender_ratio_orders_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\gender_ratio_orders_count')
dict_to_pickle(gender_ratio_visits_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\gender_ratio_visits_count')
dict_to_pickle(gender_ratio_last_visited_categories, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\gender_ratio_last_visited_categories')
dict_to_pickle(gender_ratio_site_meta_list, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\gender_ratio_site_meta_list')
dict_to_pickle(gender_ratio_item_ids_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\gender_ratio_item_ids_count')
dict_to_pickle(gender_ratio_category_path_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\gender_ratio_category_path_count')
dict_to_pickle(gender_ratio_brand_ids_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\gender_ratio_brand_ids_count')
dict_to_pickle(gender_ratio_selected_sites, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\gender_ratio_selected_sites')

end_time_block3 = timeit.default_timer()
execution_time_block3 = round(end_time_block3 - start_time_block3, 2)
print(f"Время выполнения блока кода 3: {execution_time_block3} секунд")

start_time_block4 = timeit.default_timer()
print('Получаем и записываем множества значимых элементов .....')

# создадим функцию, которая в каждом словаре найдет значимые элементы, базовые параметры от 100 присутствией
# и перекос одного из значений в 1.5 раза (вот это можно очень продробно настравивать)
def find_significant_elements(gender_stats_dict, min_interactions=100, ratio_threshold=1.5):
    significant_elements = set()
    for element, stats in gender_stats_dict.items():
        total_interactions = stats['male'] + stats['female']
        if total_interactions > min_interactions:
            if max(stats['male'], stats['female']) / max(min(stats['male'], stats['female']), 1) >= ratio_threshold:
                significant_elements.add(element)
    return significant_elements

# Применение функции к каждому из словарей gender_stats что бы получить значимые позиции
significant_orders_count = find_significant_elements(orders_count_stats)
significant_visits_count = find_significant_elements(visits_count_stats)
significant_last_visited_categories = find_significant_elements(last_visited_categories_stats)
significant_site_meta_list = find_significant_elements(site_meta_list_stats)
significant_item_ids_count = find_significant_elements(item_ids_count_stats)
significant_category_path_count = find_significant_elements(category_path_count_stats)
significant_brand_ids_count = find_significant_elements(brand_ids_count_stats)
significant_selected_sites = find_significant_elements(selected_sites_stats)

# Запишем наши множества на диск
dict_to_pickle(significant_orders_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\significant_orders_count')
dict_to_pickle(significant_visits_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\significant_visits_count')
dict_to_pickle(significant_last_visited_categories, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\significant_last_visited_categories')
dict_to_pickle(significant_site_meta_list, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\significant_site_meta_list')
dict_to_pickle(significant_item_ids_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\significant_item_ids_count')
dict_to_pickle(significant_category_path_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\significant_category_path_count')
dict_to_pickle(significant_brand_ids_count, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\significant_brand_ids_count')
dict_to_pickle(significant_selected_sites, 'C:\\Users\\HP\\Desktop\\hacaton\\grid_dict\\significant_selected_sites')

end_time_block4 = timeit.default_timer()
execution_time_block4 = round(end_time_block4 - start_time_block4, 2)
print(f"Время выполнения блока кода 4: {execution_time_block4} секунд")

print('Программа закончила свою работу необходимые для предсказания словари добавлены в папку grid_dict, для предсказания используйте файл main.py')

end_time = timeit.default_timer()
execution_time = round((end_time - start_time) / 60, 2)
print(f"Общее время выполнения блока кода: {execution_time} минут")
