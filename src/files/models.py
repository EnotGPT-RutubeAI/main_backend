from datetime import datetime

from sqlalchemy import Column, Integer, ForeignKey, String, DateTime, Boolean, ARRAY, JSON, inspect
from sqlalchemy.orm import class_mapper

from src.BaseModel import BaseModel


class User(BaseModel):
    __tablename__ = "users"

    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    phone = Column(String, nullable=False, unique=True)
    code = Column(Integer, nullable=True)
    device = Column(String, nullable=True)

    ipaddress = Column(String, nullable=True)
    code_created_at = Column(DateTime, default=datetime.utcnow, nullable=True)

    registration_code = Column(Integer)
    registration_device = Column(String)
    activate = Column(Boolean, server_default="false")


class File(BaseModel):
    __tablename__ = "files"

    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    type = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    extension = Column(String, nullable=False)
    hash = Column(String, nullable=False)

    video_name = Column(String)
    video_path = Column(String)
    audio_path = Column(String)
    description = Column(String)
    tags = Column(ARRAY(String))
    text = Column(String)
    segments = Column(JSON)
    improved_text = Column(String)
    more_interesting = Column(JSON)

    async def file_to_dict(self):
        data = {}
        mapper = inspect(self.__class__)

        for column in mapper.columns:
            column_name = column.key
            value = getattr(self, column_name)

            if isinstance(value, datetime):
                data[column_name] = value.isoformat()
            else:
                data[column_name] = value

        return data


class Recommendation(BaseModel):
    __tablename__ = "recommendations"

    original_file_id = Column(Integer, ForeignKey("files.id"), index=True, nullable=False)
    #user_id = Column(Integer, ForeignKey("users.id"), index=True)
    recommendation_file_id = Column(Integer)
    start_from_original = Column(Integer)
    end_from_original = Column(Integer)
    url = Column(String)


