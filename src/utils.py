from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import cv2
from fastapi import HTTPException

from jose import jwt

import nltk

from src.decode import verify_jwt_token

nltk.download('punkt_tab')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"


def create_jwt_token(data: dict, expires_delta: timedelta = timedelta(days=30000)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def method_protected(token):
    return await verify_jwt_token(token)


async def method_for_role(token, role: str | list):
    user_info = await method_protected(token)
    user_role = user_info['role']
    if isinstance(role, str):
        if user_role == role:
            return user_info
    elif isinstance(role, list):
        if user_role in role:
            return user_info
    raise HTTPException(status_code=401, detail="Метод недоступен для этого пользователя")


# text = "Когда я понял, что буду делать онлайн игру, я хотел сделать ее как можно быстрее, иначе я мог бы потерять хния интерес и забросить ее. Очевидное решение в данном случае использовать фраим ворки. В данном случае фраим ворк нужен был именно для ускорения работы. У меня была мало опыта работы с JavaScript фраим ворками, поэтому я выбрал тот, о котором больше всего говорили. Это был реакт-жез. Итак, я начал разрабатывать игру на реакт-жез. Но меня не покидала чувство, что мне приходится писать больше, чем стоило бы. При работе с реакт-жез, прежде всего вам нужно решить, какие компоненты вы собираетесь использовать в функциональные или классовые. Когда вы новичок, у вас нет предпочтений. И вам нужно изучать различие. Это займет некоторое время. Я как бы к антразработчику, работающий в сабъектно-орентированным языком, склоняясь к классовым компонентам. Но реакт по какой-то причине рекомендует функциональные компоненты, что для меня казалось странным. Затем нужно изучать новые концепции, например, хуки. И последний капель стал тот факт, что нужно написать относительно много кода, чтобы привязать модели к UI-элементу. Я ожидал, что могу просто диклоративно указать свойство модели, и все заработает автоматически. Но нет. Мне нужно написать обработчик для обновления модели. Итак, я обратился к Google, чтобы найти альтернативный фраимурк. Эльтернативами были Angular и VGS. Много лет назад я сталкивался с Angular, но я помню, что у него были проблемы с производительностью. Я посмотрел сравнение трех фраимурков и оказалось, что VGS был быстрее и занимал меньше места. Поэтому я решил перейти на VGS. И сразу скорость разработки увеличилась. Вам не нужно много изучать VGS, достаточно простого видения, а затем время от времени посещать Google для решения специфических проблем. Одним из недостатков в VGS, как упоминалось в некоторых статьях, было то, что он не заставлял вас писать хороший код. Но я не думаю, что это серьезный недостаток, по крайней мере, в моем случае. Если бы мне пришлось выбирать фраимурк для проекта в команде с разным уровнем опыта, я бы учитывал такой недостаток. Но для небольшого личного проекта это можно игнорировать. Я должен признать, что позже, когда игра была почти готова, я подумал, что ReactJS, возможно, был бы более лучшим выбором для реализации некоторых функций. Но только если бы я уже имел большой опыт работы с реакт. Как я уже упоминал, основной целью фраимурк было ускорение разработки. В VGS лучше справился с этой задачей, чем реакт. Другой тип фраимурка, который обычно используется для ускорения выбора разработки, это CSS фраимурк. Если бы я тщательно спланировал пользовательский интерфейс, я бы понял, что мне не нужен CSS фраимурк, но я не спланировал. Я не выбрал будд страф, так как с силей там уже устарили. Я быстро посмотрел варианты и материала CSS показался мне хорошим вариантом. Но как оказалось, мне вообще не понадобился CSS фраимурк. Я потратил больше времени на попытки реализовать пользовательский интерфейс с существующими стилями, чем если бы я использовал простой CSS. Подведем итог. Вот что вы можете из влечь из этого видео. Если вы выбираете между реакт и вью, выберите реакт, если вы хотите создать большой проект в команде. И вью, если хотите быстро сделать небольшой личный проект. Если вы делаете типовой веб-сайт, используйте CSS фраимурк. Это ускорит разработку. Если вы хотите создать кастом на интерфейс, забудьте о фраимурке и используйте чистый CSS."


def summarization(text: str, sentences_count: int = 3):
    parser = PlaintextParser.from_string(text, Tokenizer("russian"))

    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    print(summary)

    result = []
    for sentence in summary:
        result.append(sentence.__str__())

    return result


def split_text_into_fragments(text, max_fragment_length=2048):
    """
  Разбивает текст на фрагменты, каждый из которых не превышает max_fragment_length символов.
  """
    sentences = re.split(r'[.!?]+', text)

    # Инициализируйте массив фрагментов.
    fragments = []

    # Инициализируйте текущий фрагмент.
    current_fragment = ""

    # Итерируйте по предложениям.
    for sentence in sentences:
        # Если текущий фрагмент пуст, добавьте первое предложение.
        if not current_fragment:
            current_fragment = sentence
        # Если добавление предложения не превышает max_fragment_length, добавьте предложение к текущему фрагменту.
        elif len(current_fragment) + len(sentence) <= max_fragment_length:
            current_fragment += " " + sentence
        # Если добавление предложения превышает max_fragment_length, добавьте текущий фрагмент к массиву фрагментов и начните новый фрагмент с текущего предложения.
        else:
            fragments.append(current_fragment)
            current_fragment = sentence

    # Добавьте последний фрагмент к массиву фрагментов.
    if current_fragment:
        fragments.append(current_fragment)

    # Верните массив фрагментов.
    return fragments


def get_frame_at_time(video_path, time):
    cap = cv2.VideoCapture(video_path)

    seconds = int(time)
    milliseconds = int((time - seconds) * 1000)

    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000 + milliseconds)

    ret, frame = cap.read()

    cap.release()

    if ret:  # Проверка на успешность считывания кадра
        cv2.imshow("Frame at {} seconds".format(time), frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to read frame at {} seconds".format(time))


# # Пример использования
# video_path = "video.mp4"
# time = 5.5  # 5 секунд 500 миллисекунд
# frame = get_frame_at_time(video_path, time)
#
# # Отображение кадра
# cv2.imshow("Frame at {} seconds".format(time), frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
