# Хакатон 2023 - 👩 👨 КОМАНДА "EXCELLENT'S"
<img src=presentation/logo.jpg width=150px height=150px>

## Авторы
САМОЙЛОВ Павел, КАДИРОВ Михаил, КЕЛАСКИНА Елизавета, ЛЕОНТЬЕВА Ольга, ЕПИШКОВ Владислав, ПЕРЕВЕРЗЕВ Петр.

## Задача
Создание модели, предсказывающей пол пользователя (м/ж) на основе данных о его активности в Интернете. Наш простой алгоритм показал результат точности 80% на валидации и ровто столько же на закрытых тестовых данных.

## Структура репозитория
📁 **final_code** - итоговый вариант кода  
> 📑 *g_dictionary.ipynb* - полный код c предсказаниями на валидационной и тестовой выборке (тут вся логика исследования)
> 📑 *test_results.csv* - файл c предсказаниями на тестовой выборке.  
> 📑 *train.py* - собирает и обучает словари из тренировочного набора.  
> 📑 *uitls.py* - функции для парсинга данных, получения прогнозных значений.

📁 **presentation** - в ней последовательно, эволюционно, так сказать, представлены результаты нашей работы (там есть и презентация, где немного рассказывается о нас)
> 📑 *Презентация final Excellent's_Flocktory.pptx* - просим ознакомиться с файлом, в нем финальная презентация нашего решения.   
> 📑 *OUR_TEAM.pptx* - немного о членах нашей команды.

📁 **development** - содержит файлы с историей разработки моделей, в нем стоит посмотреть:
> 📑 *d_dictionary_v3.ipynb* - интересен тем, что в нем есть результаты работы словарей, обученных делать прогноз через определение доминантного пола (для сайта, товара и т.д.), и его прогнозы достигают точности 0.79. Ну и конечно там видны этапы настройки модели  
> 📑 *vector.ipynb* - в нем представлены результаты обучения моделей классического ML, основанных на вытягивании признаков в вектор, в этом мини-соревновании победила модель GB с результатом 0.78.  
> 📁 *XGBoost_val_0.77* - обучение на XGBoost (признаки агрегационные и уникальные категории товаров из general_categories_mapping, last visits categories). Содержит ноутбук с обучением и оценкой модели,
описанием признаков; скрипт, принимающий путь к файлу json на вход и выдающий предикт + модуль обработки данных; тестовые предсказания; сериализованная модель и категории товаров.

📁 **stats_dict** - содержит словари со статистикой по покупкам, посещениям сайтов мужчинами и женщинами, эти файлы могут быть полезны для аналитики.

📁 **grid_dict** - содержит 8 словарей, где рассчитаны доли посещения / покупок мужчин и женщин, а так же 8 множеств, в которых собраны значимые сайты, товары, категории. Значимыми мы считаем те, у которых более 100 взаимодействий с пользователями и есть перекос в сторону одного из полов в 1.5 раза.

📁 **analytics** - собраны ноутбуки и картинки с аналитикой, там мы увидели товары, которые покупают только мужчины, только женщины, купленные всего 1 раз и многое другое, просим ознакомиться.  
> Отдельно можно посмотреть макет в ***excel*** и поиграть с точностью модели (правда ее первой версии обученной на доминантных признаках) в зависимости от разницы мужского и женского скора и популярности сайта, товара, и категории. На этом манекене мы пришли к выводу, что если мы хотим отдавать в продакшен хорошую точность, мы должы воздержаться от прогноза когда скор в пользу женщин на 10 единиц, ну и еще к нескольким выводам, о них мы расскажем в презентации. ❗

📁 **data** - в этой папке описание категорий, полученные от заказчика (их преобразование вы можете посмотреть в папке **development** в файле **description**). Также обрезанный json, с которого и началось наше знакомство с данными, не обрезанный для знакомства грузился слишком долго. Ну и история первого пользователя из тестового набора, ее мы вырезали из основного тестового набора для отладки будущего API, и да мы предполагаем что это девушка:)

📁 **documents** - описание задания и контрольные сроки проведения хакатона.  

> 📑 *main.py* - принимает на вход строку с данными и предсказывает пол, в текущем исполнении он может принимать и набор строк.  
> 📑 *api_base.py* - заготовка для программы, которая может крутиться на сервере, заодно проверили, что код корректно обрабатывает 1 строку json.
