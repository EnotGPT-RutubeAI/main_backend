import os
from datetime import datetime

from fastapi import HTTPException
from jose import jwt

#from exceptions import FailResponse

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"


async def verify_jwt_token(token):
    token = token.credentials
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
        expiration_time = decoded_token.get("exp")
        if expiration_time:
            current_time = datetime.utcnow()
            if current_time < datetime.fromtimestamp(expiration_time):
                return decoded_token
    except jwt.ExpiredSignatureError:
        # Токен истек
        raise HTTPException(401,"Срок действия токена истёк. Обрaтитесь за новым токеном")
    except jwt.JWTError:
        raise HTTPException(401, "Неверный токен")

    # Токен недействителен
    raise HTTPException(401, "Токен недействителен")