print('Загружаем нужные библиотеки...')
import pandas as pd

from sys import argv
import timeit

from final_code.utils import test_data, predict_gender


try:
    file_path_test = argv[1]
except:
    file_path_test = input(
        'Передайте путь к json-файлу, (например, ./data/first_user.json): '
    )
    if file_path_test == '': file_path_test = './data/first_user.json'

 
df_test = pd.read_json(file_path_test)
if not len(df_test.columns) == 1:
    try:
        id_num = argv[2]
    except:
        first_id = df_test.columns[0].split('_')[1]
        last_id = df_test.columns[-1].split('_')[1]
        id_num = input(f'Введите id пользователя ({first_id} - {last_id}): ')
    col = f'user_{id_num}'
    df_test = df_test.loc[:, [col]]

print('Активируем функции для парсинга данных и выполняем парсинг...')
start_time = timeit.default_timer()

df_test = test_data(df_test)

print('Приступаем к предсказаниям...')
start_time_block1 = timeit.default_timer()

df_check_test = pd.DataFrame()
df_check_test['id'] = df_test['user_id']
# Применение функции к DataFrame
df_check_test['predict_gender'] = df_test.apply(predict_gender, axis=1)

print('Магия произошла:')
print(f'Пользователь {df_check_test.iat[0, 0]} предположительно {df_check_test.iat[0, 1]}')

end_time_block1 = timeit.default_timer()
execution_time_block1 = round(end_time_block1 - start_time_block1, 2)
print(f'Время выполнения предсказания: {execution_time_block1} секунд')

end_time = timeit.default_timer()
execution_time = round((end_time - start_time), 2)
print(f'Общее время выполнения блока кода: {execution_time} секунд')