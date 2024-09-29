# Основной сервер автоматизации создания виральных клипов

Основной backend-сервер, принимающий запросы и направляющий их по соответствующим серверам(модели тепловых карт и модели Whisper и Llama).

## Возможности

- Загрузка видео, аудио, фото
- Сбор информации о видео: теги, описание
- Сбор информации о видео: распознавание текста, получение сегментов
- Сбор информации о видео: получение данных тепловых карт

## Начало работы

### Требования

- Python >= 3.9.5

### Установка
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/EnotGPT-RutubeAI/main_backend
2. Установите необходимые пакеты:
   ```bash
   pip install -r requirements.txt
3. Запуск приложения:
   ```bash
   python main.py


### Дерево проекта

```
├── .env # чувствительные данные
├── main.py # основной запускаемый файл
├── requirements.txt # зависимости
├── tree_output.txt
├── yolov8n.pt # модель yolo8
├── database 
│   ├── __init__.py
│   ├── database.py # файл для подключения к БД
│   └── __pycache__
│       ├── __init__.cpython-39.pyc
│       └── database.cpython-39.pyc
├──src
│   ├── __init__.py
│   ├── BaseModel.py # абстрактная модель таблицы БД
│   ├── decode.py # файл для декодирования данных токена
│   ├── exceptions.py 
│   ├── utils.py # Прочее
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc   
│   │   ├── BaseModel.cpython-39.pyc
│   │   ├── decode.cpython-39.pyc
│   │   ├── exceptions.cpython-39.pyc
│   │   └── utils.cpython-39.pyc
│   └── files # основной модуль этого сервера
│       ├── __init__.py
│       ├── models.py # модели таблиц БД
│       ├── router.py # маршруты сервера
│       ├── schemas.py # схемы данных для ответов и запросов
│       ├── service.py # основной функционал
│       └── __pycache__
└── uploads # папки с файлами
   ├── audios
   ├── documents
   ├── photos
   └── videos
```
