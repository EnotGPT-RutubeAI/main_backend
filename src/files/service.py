import hashlib
import heapq
import json
import os
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher

import cv2
import numpy as np
from requests_async import post
# import requests
from fastapi import UploadFile, File, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from moviepy.video.io.VideoFileClip import VideoFileClip
from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse
from ultralytics import YOLO

from src.exceptions import SuccessResponse
from src.files.models import File as FileModel, Recommendation

from database.database import get_db
from src.decode import verify_jwt_token
from src.utils import summarization, split_text_into_fragments, get_frame_at_time

security = HTTPBearer()

ALLOWED_EXTENSIONS_PHOTO = (".jpg", ".jpeg", ".png")
ALLOWED_EXTENSIONS_VIDEO = (".mp4", ".avi", ".mov")
ALLOWED_EXTENSIONS_DOCUMENT = (".pdf", ".txt")
ALLOWED_EXTENSIONS_AUDIO = ("wav", "mp3")

MAX_PHOTO_FILE_SIZE = 45 * 1024 * 1024  # 45 мегабутов
MAX_VIDEO_FILE_SIZE = 250 * 1024 * 1024  # 250 мегабутов
MAX_DOCUMENT_FILE_SIZE = 20 * 1024 * 1024  # 250 мегабутов
MAX_AUDIO_FILE_SIZE = 45 * 1024 * 1024

PATH_PHOTO = f"{os.getenv('UPLOADS')}/photos"
PATH_VIDEO = f"{os.getenv('UPLOADS')}/videos"
PATH_DOCUMENT = f"{os.getenv('UPLOADS')}/documents"
PATH_AUDIO = f"{os.getenv('UPLOADS')}/audios"

FILE_TYPES = ("photo", "video", "document", "audio")

types_extensions = {
    "photo": ALLOWED_EXTENSIONS_PHOTO,
    "video": ALLOWED_EXTENSIONS_VIDEO,
    "document": ALLOWED_EXTENSIONS_DOCUMENT,
    "audio": ALLOWED_EXTENSIONS_AUDIO
}

types_sizes = {
    "photo": MAX_PHOTO_FILE_SIZE,
    "video": MAX_VIDEO_FILE_SIZE,
    "document": MAX_DOCUMENT_FILE_SIZE,
    "audio": MAX_AUDIO_FILE_SIZE
}

types_paths = {
    "photo": PATH_PHOTO,
    "video": PATH_VIDEO,
    "document": PATH_DOCUMENT,
    "audio": PATH_AUDIO
}


class WhisperLlamaData:
    """
    Данные, пришедшие из виспера
    """
    def __init__(self, name: str, description: str, tags: list, interesting_moments: list):
        self.name = name
        self.description = description
        self.tags = tags.copy()
        self.interesting_moments = interesting_moments.copy()


def allowed_file(filename: str, allowed_extensions: tuple) -> bool:
    """Проверяет расширение файла."""
    return any(filename.endswith(ext) for ext in allowed_extensions)


def secure_filename(filename: str) -> str:
    """Возвращает безопасное имя файла с захешированным названием."""
    hashed_name = hashlib.sha256(filename.encode() + datetime.now().strftime("%Y%m%d%H%M%S%f").encode()).hexdigest()
    ext = os.path.splitext(filename)[1]
    return f"{hashed_name[:30]}{ext}"


def get_extension(filename: str) -> str:
    """Возвращает все, что после точки в строке."""
    if '.' in filename:
        return filename.split('.')[-1]  # Возвращаем последний элемент после точки
    return ''  # Если точки нет, возвращаем пустую строку


def extract_name(filename):
    """
    Извлечь хеш из название.wav
    :param filename:
    :return:
    """
    last_dot_index = filename.rfind('.')
    name = filename[:last_dot_index] if last_dot_index != -1 else filename
    return name


def get_filename_without_extension(filename: str) -> str:
    """
    Возвращает имя файла без расширения.

    :param filename: Имя файла.
    :return: Имя файла без расширения.
    """
    filename, extension = os.path.splitext(filename)
    return filename


async def upload(file_type: str, request: Request,
                 file: UploadFile = File(...),
                 token: HTTPAuthorizationCredentials = Depends(security),
                 db: AsyncSession = Depends(get_db)):
    """
    Загрузка любого типа файла
    :param file_type:
    :param request:
    :param file:
    :param token:
    :param db:
    :return:
    """
    token = await verify_jwt_token(token)
    user_id = token['id']

    if not allowed_file(file.filename, types_extensions[file_type]):
        raise HTTPException(status_code=400,
                            detail=f"Неверное расширение файла. Разрешены только {str(types_extensions[file_type])}")

    file_size = await file.read()
    if len(file_size) > types_sizes[file_type]:
        raise HTTPException(status_code=400, detail="Размер файла превышает допустимый лимит.")

    safe_name = secure_filename(file.filename)

    upload_path = os.path.join(types_paths[file_type], safe_name)

    with open(upload_path, "wb") as buffer:
        buffer.write(file_size)

    upload_file = FileModel(user_id=user_id,
                            type=file_type,
                            name=file.filename,
                            hash=safe_name,
                            extension=get_extension(safe_name))
    db.add(upload_file)
    await db.commit()
    scheme = os.getenv("DOMAIN")
    return SuccessResponse({"url": os.getenv("UPLOADS") + f"videos/{safe_name}"})


async def get_audio_recognition(audio_url: str):
    """
    Получает данные виспера
    :param audio_url:
    :return:
    """
    url = os.getenv("WHISPER_SERVER") + "audio_to_text"
    response = await post(url=url, json={"audio_url": audio_url})
    print(response.text)
    return response.json()


def extract_filename(url) -> str:
    """
    Обрезает из ссылки в хеш
    :param url:
    :return:
    """
    last_slash_index = url.rfind('/')
    filename = url[last_slash_index + 1:]
    return filename


async def write_recognition_data(db: AsyncSession, audio_url: str, extension: str, data: dict) -> dict:
    """
    Записывает распознанные данные в файл
    :param extension:
    :param db:
    :param audio_url:
    :param data:
    :return:
    """
    audio_hash = extract_name(extract_filename(audio_url)) + '.' + extension
    print(audio_hash)
    upd = await db.execute(update(FileModel).where(FileModel.hash == audio_hash)
                           .values(text=data['text'], segments=data['chunks']))
    await db.commit()
    return data


async def request_to_llm(url: str, messages: list):
    """
    Отправляет запрос ламе
    :param url:
    :param text:
    :param its_first:
    :return:
    """
    response = await post(url, json={"data": messages})
    print(response.text)
    print(clear_json(response.text)['response'])
    return clear_json(response.text)['response']


def clear_json(json_str: str):
    '''
    Возврат json строки из ответа
    '''
    json_str = json_str[json_str.find("{"):json_str.rfind("}") + 1]
    return json.loads(json_str)


def insert_last_symbol_in_json(text: str):
    """
    Доопределяем форму json
    :param text:
    :return:
    """
    if text[-1] != "}":
        text = text + "}"
    return text


async def get_big_interesting_moments(text: str):
    """
    Запрос на получение цельных интересных сегментов
    :param text:
    :return:
    """
    url = os.getenv("LLM_SERVER") + "llama_ollama"
    # system_prompt = ("Из текста нужно найти самые интересные и самые важные моменты видео. Те, которые могут заинтересовать."
    #         "Твоя задача вернуть мне json, в котором будет одно поле: moments - это будет массив цельных фрагментов"
    #         " из текста. Именно завершенных. Таких, которые смогут заинтересовать читателя."
    #         "Если большой текст - нужно больше фрагментов, если малый - пусть будет немного. "
    #         "Главное - интересность и значимость фрагментов. "
    #         "Для не малого текста сделай от 4 до 15 фрагментов, отсортируй их по интересности(важности или значимости).")

    messages = []
    messages.append({"role": "system",
                     "content": "Из текста нужно найти самые интересные моменты для человека, самые ключевые, самые важные. "
                                "Твоя задача отправить JSON следующего вида по моему тексту: "
                                "json``` {\"moments\": [<момент_1>, <момент_3>, <момент_3>]}\\'``` "
                                "\nОчень важно, чтобы моменты были теми, которые могут заинтересовать. Они должны быть цельными. "
                                "Один элемент массива желательно чтобы был не менее 150 символов и не более 600 символов."
                                "Обязательно сохрани структуру JSON, которую я тебе описал"

                     })
    text_segments = split_text_into_fragments(text)
    for text_segment in text_segments:
        print(text_segment)
        messages.append({"role": "user", "content": text_segment})

    messages.append({"role": "user",
                     "content": "Из текста нужно найти самые интересные моменты для человека, самые ключевые, самые важные. "
                                "Твоя задача отправить JSON следующего вида по моему тексту: "
                                "json``` {\"moments\": [<момент_1>, <момент_3>, <момент_3>]}\\'``` "
                                "\nОчень важно, чтобы моменты были теми, которые могут заинтересовать. Они должны быть цельными. "
                                "Один элемент массива желательно чтобы был не менее 150 символов и не более 600 символов(если текст не маленький)."
                                "Обязательно сохрани структуру JSON, которую я тебе описал и больше ничего не пиши. И пусть моментов будет побольше, если текст большой!"
                     })

    result = await request_to_llm(url, messages)
    if type(result) == type(str()):
        result = insert_last_symbol_in_json(result)
        result = clear_json(result)
        # result = json.loads(result)
    return result


async def get_llama_info(segments: list):
    """
    Выполняем обращение к ллама севреру
    :param segments:
    :param segment_text:
    :param request_text:
    :return:
    """
    # 2048 токенов

    url = os.getenv("LLM_SERVER") + "llama_ollama"
    # text = "Ответ верни в формате JSON. "
    # text += "В JSON пусть будут поля: name, description, tags, interesting_moments. "
    # text += "name - название, которое бы ты дал видео по этому тексту. description - краткое описание. "
    # text += "Описание пиши так, будто это реальное описание на видеохостинге. "
    # text += "interesting_moments - самый важный массив должен быть. Туда запиши те элементы из текста, "
    # text += "которые могут заинтересовать человека наиболее всего. То есть выпиши наиболее заинтересовывающие моменты(сегменты, выделенные '')"
    # text += "и отсортируй их от самого интересного до менее интересных. Текст будет в сегментах. Текст: "

    # text = ("Я тебе отправил текст в предыдущих сообщениях. "
    #         "твоя задача вернуть мне json, где будет 4 поля: name, description, tags, interesting_moments. "
    #         "В названии укажи, как бы ты назвал видео с таким текстом внутри, в описании - краткое описание виде. "
    #         "Например: автор повествует, главный герой борется за... и так далее. "
    #         "Только не пиши, пожалуйста, что-то по типу: в этом видео вы увидите... "
    #         "В тегах - 3 тега для видео. А вот в interesting_moments верни те строки, которые прямо влияют "
    #         "на интерес, которые способны заинтересовать посмотреть видео с этим текстом. "
    #         "То есть верни именно те, которыми можно заинтересовать. Слово в слово. И отсортируй их по "
    #         "убыванию возможности заинтересовать. Текст я тебе дам по сегментам, в массив занеси самые "
    #         "интересные сегменты с точки зрения их возможности заинтересовать. "
    #         "Отсортируй этот массив по убыванию интересности.")

    # text_2 = ("Я тебе отправлю текст. "
    #         "твоя задача вернуть мне json, где будет 1 поле: more_interesting. "
    #         "В more_interesting верни те строки, которые прямо влияют "
    #         "на интерес, которые способны заинтересовать посмотреть видео с этим текстом. "
    #         "То есть верни именно те, которыми можно заинтересовать. Слово в слово. И отсортируй их по "
    #         "убыванию возможности заинтересовать. Текст я тебе дам по сегментам, в массив занеси самые "
    #         "интересные сегменты с точки зрения их возможности заинтересовать. "
    #         "Отсортируй этот массив по убыванию интересности.")

    # text_2 = "Ответ верни в формате JSON. В JSON пусть будет поле interesting_moments. "
    # text_2 += "это поле пусть будет массивом. Туда запиши те элементы текста ниже, которые могут заинтересовать "
    # text_2 += "человека больше всего. То есть выпиши наиболее заинтересовывающие моменты "
    # text_2 += "и отсортируй их от самого интересного до менее интересных. Текст будет в сегментах. Текст: "

    messages = []
    messages.append({"role": "system",
                     "content": "Из текста нужно найти самые интересные моменты для человека. Твоя задача вернуть json такого вида по моему тексту: json``` {\"name\": \"<название>\",\"description\": \"<описание>\",\"tags\": [\"<тег1>\", \"<тег2>\"],\"interesting_moments\": [\"<интересный_момент1_дословно>\",\"<интересный_момент2_дословно>\",\"<интересный_момент3_дословно>\"]}\\'``` .В поле interesting_moments пиши слово в слово из текста целое предложение. Интересных оментов должно быть не много, но они должны быть реально заинтересовывающими. В остальных полях(name, tags, description) пиши то, что считаешь нужным, как это могло бы выглядеть на реальном видеохостинге. Кроме JSON никакого текста не пиши. JSON обязательно должен быть валидным"
                     })

    # content = ""
    # all_text = ""
    # for segment in segments:
    #     all_text += "'" + segment + "', "
    # all_text = all_text[:-2]
    # messages.append({"role": "user", "content": all_text})
    # response = await request_to_llm(url, messages)
    # print(response.text)
    # return messages
    content = ""

    for segment in segments:
        if len(content + segment) <= 2048:
            content += segment
        else:
            messages.append({"role": "user", "content": content})
            # print(summarization(content))
            content = segment
    messages.append({"role": "user", "content": content})
    print(messages)
    # print(summarization(content))
    result = await request_to_llm(url, messages)
    if type(result) == type(str()):
        result = insert_last_symbol_in_json(result)
        result = json.loads(result)
    return result

    # len_text = len(text)
    # segments_text = ""
    # its_first = True
    # all_data = []
    # for segment in segments:
    #     print(segment)
    #     segments_text += "'" + segment + "', "
    #     if len_text + len(segments_text) <= 2048:
    #         text += segments_text
    #         len_text = len(text)
    #     else:
    #         text = text[:-2]
    #         result = await request_to_llm(url, text, its_first)
    #         all_data.append(result)
    #         its_first = False
    #         text = text_2
    #         len_text = len(text)
    # if its_first:
    #     text = text[:-2]
    #     result = await request_to_llm(url, text, its_first)
    #     all_data.append(result)
    # print(all_data)
    # тут надо обработать all_data
    # return {"response": all_data}


def delete_segment_symbols(segment: str):
    if segment.startswith("'"):
        text = segment[1:]

    if segment.endswith("'"):
        segment = segment[:-1]
    return segment


async def get_segments(data: dict) -> list:
    """
    Из респонса распознания виспера выделяем сегменты
    :param data:
    :return:
    """
    segments = []
    for chunk in data['chunks']:
        segments.append(chunk['text'])
    return segments


async def update_file_models_data(db: AsyncSession, file_id: int, name: str, description: str, tags: list,
                                  segments: dict, more_interesting: list):
    upd = await db.execute(update(FileModel).where(FileModel.id == file_id)
                           .values(name=name, description=description, tags=tags,
                                   segments=segments, more_interesting=more_interesting))
    await db.commit()
    return file_id


async def upload_video(file_type: str,
                       file: UploadFile = File(...),
                       token: HTTPAuthorizationCredentials = Depends(security),
                       db: AsyncSession = Depends(get_db)):
    token = await verify_jwt_token(token)
    user_id = token['id']

    if not allowed_file(file.filename, types_extensions[file_type]):
        raise HTTPException(status_code=400,
                            detail=f"Неверное расширение файла. Разрешены только {str(types_extensions[file_type])}")

    file_size = await file.read()
    if len(file_size) > types_sizes[file_type]:
        raise HTTPException(status_code=400, detail="Размер файла превышает допустимый лимит.")

    safe_name = secure_filename(file.filename)

    upload_path = os.path.join(types_paths[file_type], safe_name)

    with open(upload_path, "wb") as buffer:
        buffer.write(file_size)

    extension = get_extension(safe_name)

    upload_file = FileModel(user_id=user_id,
                            type=file_type,
                            name=file.filename,
                            hash=safe_name,
                            extension=get_extension(safe_name))

    db.add(upload_file)
    await db.commit()
    await db.refresh(upload_file)
    file_id = upload_file.id
    scheme = os.getenv("DOMAIN")
    url = scheme + f"{file_type}s/{safe_name}"

    result = {"response": {}}
    result['file'] = {}
    # сохраняем аудио в url
    audio_url = await get_audio_url(safe_name)
    result['file']['id'] = file_id
    result['file']['url'] = await get_video_url(safe_name)
    print(audio_url)
    # прогоняем через виспер
    response = await get_audio_recognition(audio_url)

    result['response']['whisper'] = response
    text = response['text']
    #

    result['big_interesting'] = await get_big_interesting_moments(text)

    # добавляем в БД данные
    data = await write_recognition_data(db, audio_url, extension, response)
    # прогоняем через ламу
    segments = await get_segments(data)
    info = await get_llama_info(segments)
    result['response']['llama'] = info
    whisper_llama_data = WhisperLlamaData(info['name'], info['description'], info['tags'],
                                          info['interesting_moments'])
    # ставим в соответствие сегментам
    llama_recommendations = await find_chunks(data['chunks'], whisper_llama_data.interesting_moments)
    llama_big_recommendations = await find_chunks(data['chunks'], result['big_interesting']['moments'])




    result['big_interesting_segments'] = llama_big_recommendations
    result['response']['llama']['recommendations'] = llama_recommendations
    # дозаписываем данные по файлу
    await update_file_models_data(db, file_id, whisper_llama_data.name, whisper_llama_data.description,
                                  whisper_llama_data.tags, data['chunks'], llama_recommendations)
    # прогоняем через моделиван
    modelivans = await get_modelivans_data(segments, response, audio_url, video_url=await get_video_url(safe_name))
    result['response']['modelivana'] = modelivans

    top_segments_modelivans = await return_top_segments_modelivans(result['response']['modelivana']['audio']['map'],
                                                                   result['response']['modelivana']['volume'][
                                                                       'deviations'],
                                                                   result['response']['modelivana']['video'],
                                                                   data['chunks'], safe_name)
    result['response']['modelivana']['top_segments'] = top_segments_modelivans


    return SuccessResponse(data=result)


async def get_modelivana_video_data(video_url: str, data: dict) -> list:
    """
    Возвращает смены кадров посекундно
    :param video_url:
    :return:
    """
    url = os.getenv("MODELIVANA_SERVER") + "video"
    response = await post(url, json={"video_url": video_url, "data": data['chunks']})
    print(response.text)
    return json.loads(response.text)['array']


async def get_modelivans_data(segments: list, whisper_data: dict, audio_url: str, video_url: str):
    # отправляем на моделивану аудио
    data = {}
    modelivana_audio = await get_modelivana_audio_data(segments)
    data['audio'] = modelivana_audio
    # тут нужно отправить на моделивану звука
    modelivana_volume = await get_modelivana_volume_data(whisper_data, audio_url)
    data['volume'] = modelivana_volume
    modelivana_video = await get_modelivana_video_data(video_url, {"chunks": whisper_data})
    data['video'] = modelivana_video

    return data


def get_top_n_max_elements(all_results, N):
    """
    Получить N наиболее высоких элементов
    :param all_results:
    :param N:
    :return:
    """
    return heapq.nlargest(N, all_results)


def get_top_n_indices(all_results, N):
    """
    Получить индексы наиболее высоких элементов
    :param all_results:
    :param N:
    :return:
    """
    return np.argsort(all_results)[-N:][::-1]


def count_percentage_elements(all_results, percentage):
    total_elements = len(all_results)
    percentage_count = round((percentage / 100) * total_elements)
    return percentage_count


async def return_top_segments_modelivans(emotions: list, volumes: list, videos: list, segments: list, filename: str,
                                         percent: int = 15):
    """
    Возвращает 15% лучших сегментов
    :param percent:
    :param emotions:
    :param volumes:
    :param segments:
    :return:
    """

    all_result = []
    # считаем сумму всех элементов
    for i in range(len(emotions)):
        # print(emotions[i], volumes[i])
        all_result.append(emotions[i] + volumes[i] + videos[i])

    count = count_percentage_elements(all_result, percent)
    if count == 0:
        count = 1

    top_indices = get_top_n_indices(all_result, count)

    result = []
    for i in top_indices:
        result.append(segments[i])
        print(segments[i])

    path = os.getenv("UPLOADS") + "videos/" + filename
    print(path)
    print("Метод YOLO Биби пошел")
    for element in result:
        pass
        # classes = get_classes_count(path, int(element['timestamp'][0]), int(element['timestamp'][1]))
        # element['classes'] = classes
    return result


async def get_modelivana_audio_data(data: list, parameter: str = None):
    """
    Получаем данные моделиваны эмоций
    :param data:
    :param parameter:
    :return:
    """
    url = os.getenv("MODELIVANA_SERVER") + "audio/emotions"
    response = await post(url, json={"data": data, parameter: None})
    print(json.loads(response.text))
    return json.loads(response.text)


async def get_modelivana_volume_data(whisper_data: dict, audio_url: str):
    """
    Получаем данные моделиваны звука
    :param whisper_data:
    :param audio_url:
    :return:
    """
    url = os.getenv("MODELIVANA_SERVER") + "audio/volume"
    response = await post(url, json={"data": whisper_data, "audio_url": audio_url})
    print(response.text)
    print(json.loads(response.text))
    return json.loads(response.text)


async def find_chunks(chunks: list, moments: list):
    result = []

    for moment in moments:
        # Находим наиболее похожий chunk
        most_similar_chunk = None
        highest_similarity = 0

        for chunk in chunks:
            similarity = SequenceMatcher(None, moment, chunk["text"]).ratio()

            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_chunk = chunk

        # Добавляем найденный chunk в результат
        if most_similar_chunk:
            result.append({"text": moment, "timestamp": most_similar_chunk["timestamp"]})
    return result


async def get_audio_url(video_hash: str) -> str:
    """
    Делает аудио из видео, отдает ссылку на аудио
    :param extension:
    :param video_hash:
    :return:
    """
    filename = get_filename_without_extension(video_hash)
    clip = VideoFileClip(os.getenv("UPLOADS") + "/videos/" + video_hash)
    audio_clip = clip.audio
    audio_clip.write_audiofile(os.getenv("UPLOADS") + "/audios/" + filename + ".mp3")
    audio_url = filename + ".mp3"
    return os.getenv("FILE_SERVER") + 'audios/' + audio_url


async def get_video_url(video_hash: str) -> str:
    """
    Получить видео url
    :param video_hash:
    :return:
    """
    filename = get_filename_without_extension(video_hash)
    video_url = filename + ".mp4"
    return os.getenv("FILE_SERVER") + 'videos/' + video_url


async def get_file(file_type: str, hash_name: str, db: AsyncSession):
    if file_type not in FILE_TYPES:
        raise HTTPException(403, "Ошибка разработчика. Чепуху написал")

    # file = await FileModel.query_one(db, select(FileModel).where(FileModel.hash == hash_name,
    #                                                        FileModel.type == file_type,
    #                                                        FileModel.is_deleted == False).limit(1))
    # if not file:
    #     raise HTTPException(404, "Файл не найден")

    return FileResponse(types_paths[file_type] + f"/{hash_name}")


def construct_full_url(request: Request, file_type: str, safe_name: str) -> str:
    """Формирует полный URL для указанного file_type и safe_name."""
    scheme = request.url.scheme
    host = request.url.hostname
    port = request.url.port

    if port:
        full_domain = f"{scheme}://{host}:{port}"
    else:
        full_domain = f"{scheme}://{host}"

    return f"{full_domain}/{file_type}s/{safe_name}"


async def getMyFiles(request: Request, file_type: str, limit: int = 10, offset: int = 0,
                     token: HTTPAuthorizationCredentials = Depends(security),
                     db: AsyncSession = Depends(get_db)):
    if file_type not in FILE_TYPES:
        raise HTTPException(404, "Нет такого типа файлов")
    token = await verify_jwt_token(token)
    user_id: int = token['id']

    files = await FileModel.query(db, select(FileModel.hash, FileModel.extension)
                                  .where(FileModel.user_id == user_id, FileModel.is_deleted == False,
                                         FileModel.type == file_type)
                                  .limit(limit).offset(offset))
    result = []
    for file in files:
        result.append({"url": construct_full_url(request, file_type, file[0]), "extension": file[1]})
    print(result)
    return SuccessResponse({"response": result})


async def bind_recommendation(db: AsyncSession, video_url: str, original_id: int, private_key: str):
    """
    Привязываем рекомендацию к оригинальному видео
    :param private_key:
    :param db:
    :param video_url:
    :param original_id:
    :return:
    """
    if os.getenv("PRIVATE_RECOMMENDATION_KEY") != private_key:
        raise HTTPException(401, "Не валидный private_key")
    recommendation = await Recommendation.query_one(db, select(Recommendation).where(Recommendation.url == video_url))
    if recommendation:
        raise HTTPException(403, "Уже привязано")
    rec = Recommendation(original_file_id=original_id, url=video_url)
    db.add(rec)
    await db.commit()
    await db.refresh(rec)
    return SuccessResponse()


def get_frame_number(cap, seconds):
    """
    Функция для получения номера кадра по времени в секундах
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ret, frame = cap.read()
    return ret, frame



def get_frame_number_for_yolo(cap, seconds):
    """
    Функция для получения номера кадра по времени в секундах
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ret, frame = cap.read()
    return int(cap.get(cv2.CAP_PROP_POS_FRAMES))

async def get_frame_by_millisecond(audio_url: str, seconds: float, db: AsyncSession):
    # video_path = "uploads/videos/f2ad06269820c1142eff1a4d5a5887.mp4"
    hash = extract_filename(audio_url)
    video_path = os.getenv("UPLOADS") + "videos/" + hash
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    ret, frame = get_frame_number(cap, seconds)

    if ret:
        cv2.waitKey(0)
        cv2.imwrite(os.getenv("UPLOADS") + "photos/{}{}.jpg".format(hash[0:17], seconds), frame)
        cv2.destroyAllWindows()
    else:
        print("Failed to read frame at {} seconds".format(seconds))

    cap.release()
    return SuccessResponse({"url": os.getenv("FILE_SERVER") + "photos/" + hash[0:17] + f"{seconds}" + ".jpg"})


def get_classes_count(path, start_segment, end_segment, confidence_threshold=0.6) -> list:
    print(start_segment)
    model = YOLO('yolov8n.pt')
    result = []
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = start_segment
    end_time = end_segment
    start_frame = get_frame_number_for_yolo(cap, start_time)
    end_frame = get_frame_number_for_yolo(cap, end_time)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = 0
    step = 2

    my_set = set()
    while cap.isOpened() and start_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            results = model(frame, verbose=False)

            for result in results:
                boxes = result.boxes
                class_counts = defaultdict(int)

                for box in boxes:
                    c = int(box.cls)
                    conf = box.conf.item()
                    if conf >= confidence_threshold:
                        class_name = model.names[c]
                        class_counts[class_name] += 1

                for class_name, count in class_counts.items():
                    my_set.add(f"{class_name} {count}")
                result = list(my_set)

        start_frame += 1
        frame_count += 1

    cap.release()
    return result


async def get_video_info(db: AsyncSession, video_hash: str):
    """
    Получить файл по hash
    :param db:
    :param video_hash:
    :return:
    """
    file = await FileModel.query_one(db, select(FileModel).where(FileModel.hash == video_hash))
    if not file:
        raise HTTPException(404, "Файл не найден")
    print(file.name)
    return SuccessResponse({"data": await file.file_to_dict()})
