# Модифицированный RawNet3 для Определения Эмоций в Аудиосигналах

Этот проект представляет собой модифицированную версию сети RawNet3, предназначенную для задачи классификации эмоций в аудиозаписях. Модель разработана для обработки данных с использованием гибридного подхода, который сочетает сверточные методы и механизмы самовнимания для извлечения эмоциональных паттернов в голосе.

Удалось превысить точность гораздо более сложной модели HuBert Large:

Precision: 0.91

Recall: 0.90

F1 score: 0.91

Ссылка на веса: https://disk.yandex.ru/d/BCrmCJ4TI1iQpg

При том, что размер модели гораздо меньше и она работает с raw аудио данными

## Датасет

Использован датасет Dusha, который содержит аудиозаписи с разными эмоциональными состояниями:

angry
neutral
other
positive
sad

## Архитектура Модели

### Модифицированная версия RawNet3:

- Замена стандартного фронтенда на MultiphaseGammatoneFB
- **Позиционное кодирование**
- Замена ReLU на GELU
- **PreEmphasis**
- Поддержка фильтров произвольного порядка
- Обучаемые коэффициенты фильтра
- **AFMS (Adaptive Feature Map Scaling)**  
- Адаптивное масштабирование признаков

## Производительность
- Компактный размер подходит для устройств с ограниченными ресурсами

## Преимущества
- Точность на уровне тяжеловесных моделей
- Высокая скорость работы

## Использование
```python
model = RawNet3.MainModel(
   nOut=256,
   encoder_type="ECA",
   sinc_stride=3,
   max_frame=200,
   sr=16000
)