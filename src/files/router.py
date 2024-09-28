import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, UploadFile, File, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from src.files import service

file_router = APIRouter()

security = HTTPBearer()

executor = ThreadPoolExecutor(max_workers=10)


@file_router.post("/photo/upload", summary="Загрузка фотографии")
async def photo_upload(request: Request, file: UploadFile = File(...),
                       token: HTTPAuthorizationCredentials = Depends(security),
                       db: AsyncSession = Depends(get_db)):
    result = await service.upload("photo", request, file, token, db)
    return result


@file_router.post("/video/upload_main_video", summary="Загрузка основного видео")
async def video_upload(file: UploadFile = File(...),
                       token: HTTPAuthorizationCredentials = Depends(security),
                       db: AsyncSession = Depends(get_db)):
    # loop = asyncio.get_event_loop()
    result = await service.upload_video("video", file, token, db)
    return result


@file_router.post("/video/upload_recommendation", summary="Загрузить рекомендательные видео")
async def video_upload(request: Request, file: UploadFile = File(...),
                       token: HTTPAuthorizationCredentials = Depends(security),
                       db: AsyncSession = Depends(get_db)):
    result = await service.upload("video", request, file, token, db)
    return result


@file_router.get("/bindRecommendationToVideo", summary="Привязать  созданную рекомендацию к оригиналу")
async def bind_recommendation(request: Request, video_url: str,
                              original_id: int, private_key: str,
                              db: AsyncSession = Depends(get_db)):

    result = await service.bind_recommendation(db, video_url, original_id, private_key)
    return result


@file_router.post("/document/upload", summary="Загрузка документа")
async def photo_upload(request: Request, file: UploadFile = File(...),
                       token: HTTPAuthorizationCredentials = Depends(security),
                       db: AsyncSession = Depends(get_db)):
    result = await service.upload("document", request, file, token, db)
    return result


@file_router.post("/audio/upload", summary="Загрузка аудио")
async def photo_upload(request: Request, file: UploadFile = File(...),
                       token: HTTPAuthorizationCredentials = Depends(security),
                       db: AsyncSession = Depends(get_db)):
    result = await service.upload("audio", request, file, token, db)
    return result


@file_router.get("/photos/{hash_name}", summary="Получить фото")
async def photos_get(hash_name: str, db: AsyncSession = Depends(get_db)):
    result = await service.get_file("photo", hash_name, db)
    return result


@file_router.get("/videos/{hash_name}")
async def photos_get(hash_name: str, db: AsyncSession = Depends(get_db)):
    result = await service.get_file("video", hash_name, db)
    return result


@file_router.get("/documents/{hash_name}")
async def photos_get(hash_name: str, db: AsyncSession = Depends(get_db)):
    result = await service.get_file("document", hash_name, db)
    return result


@file_router.get("/audios/{hash_name}")
async def photos_get(hash_name: str, db: AsyncSession = Depends(get_db)):
    result = await service.get_file("audio", hash_name, db)
    return result


@file_router.get("/getMyFiles", summary="Получение списка моих загруженных видео")
async def get_my_files(request: Request,
                       type: str,
                       limit: int = 10, offset: int = 0,
                       token: HTTPAuthorizationCredentials = Depends(security),
                       db: AsyncSession = Depends(get_db)):
    result = await service.getMyFiles(request, type, limit, offset, token, db)
    return result


@file_router.get("/getFrameBySecond", summary="Получить фрейм по миллисекунде")
async def get_frame_by_millisecond(audio_url: str, seconds: float, db: AsyncSession=Depends(get_db)):
    return await service.get_frame_by_millisecond(audio_url, seconds, db)


@file_router.get("/getVideoInfo", summary="Получить информацию по загруженному видео")
async def get_video_info(video_hash: str,  db:AsyncSession=Depends(get_db)):
    return await service.get_video_info(db, video_hash)
