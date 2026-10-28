from src.DataProcessor import DataProcessor

def main():
    processor = DataProcessor()

    '''Получаем датафреймы'''
    prefiltered_df = processor.get_afisha_df()
    rub_tenge = processor.get_currency_df()
    prefiltered_strings = prefiltered_df.shape[0]

    '''Обрабатываем датафрейм Афиши'''
    df = processor.preprocessing(prefiltered_df, rub_tenge)
    postfiltered_strings = df.shape[0]


    print(f"Всего отфильтровано {round((1-postfiltered_strings / prefiltered_strings) * 100, 2)}% данных")

    '''Создаём профили пользователей'''
    user_stats = processor.create_user_stats_df()

    '''Сегментируем пользователей'''
    segmentation_device = processor.device_segmentation(user_stats)
    segmentation_event = processor.event_segmentation(user_stats)
    segmentation_region = processor.region_segmentation(user_stats)

    '''Рассчитываем Retention и визуализируем'''
    processor.display_device_retention(segmentation_device)
    processor.display_event_retention(segmentation_event)
    processor.display_region_retention(segmentation_region)

    '''Проводим корреляционный анализ и визуализируем'''
    processor.corr_analysis(user_stats)

    '''Печатаем вывод'''
    print('''
В ходе исследования данные подверглись обработке.
* Были удалены данные, для которых количество билетов превышало 99-ый перцентиль.
* Были удалены заказы с отрицательной ценой.
* Были отфильтрованы заказы, для которых цена превышала 99-ый перцентиль.
* Цена заказа была приведена к рублям.
* Для лучшей аналитики данные были сгруппированы для выделения характеристик пользователей.

Всего в выборке присутствует информация о 21392 пользователях, наибольшая часть из которых сделала меньше 5 заказов. Большое внимание уделено цене заказа и среднему времени между заказами. На основании полученной информации, они больше всего влияют на возвращение клиента, однако среднее время между заказами влияет именно на вернувшихся клиентов для оценки будущих возвращений. Также, для мониторинга возвращения клиента после повторного заказа следует обратить внимание на средний интервал между заказами, так как можно оценить, когда клиент вернётся снова.

Больше всего пользователей возвращается в данных сегментах:
* Сделавшие первый заказ с компьютера
* Чей первый заказ был на выставку или театр
* Сделавшие первый заказ в Шанырском регионе и Светополянском округе
    ''')

if __name__ == "__main__":
    main()