from typing import List, Optional, Tuple, TypeVar
from datetime import datetime
from sqlalchemy import Column, Integer, DateTime, Boolean, select, Result, Executable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.roles import _T_co

from database.database import Base

class BaseModel(Base):
    """
    Асинхронная абстрактная модель для FastAPI.
    """

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted = Column(Boolean, nullable=False, default=False)

    async def to_dict(self, hidden_fields: Optional[List[str]] = None) -> dict:
        """
        Преобразует модель в словарь, исключая скрытые поля.
        """
        data = self.__dict__.copy()
        if hidden_fields:
            for field in hidden_fields:
                data.pop(field, None)
        return data

    @classmethod
    async def get_by_id(cls, db_session: AsyncSession, model_id: int) -> Optional["BaseModel"]:
        """
        Возвращает модель по ее ID.
        """
        query = select(cls).where(cls.id == model_id)
        result = await db_session.execute(query)
        return result.scalar()

    @classmethod
    async def get_all(cls, db_session: AsyncSession, offset: int = 0, limit: int = 100):
        """
        Возвращает список моделей с учетом смещения и ограничения.
        """
        result = await db_session.execute(select(cls).offset(offset).limit(limit))
        result = result.fetchall()
        return result

    @classmethod
    async def query(cls, db_session: AsyncSession, query: Executable):
        """
        Возвращает список моделей с учетом смещения и ограничения.
        """
        result = await db_session.execute(query)
        result = result.fetchall()
        return result

    @classmethod
    async def query_one(cls, db_session: AsyncSession, query: Executable):
        """
        Возвращает список моделей с учетом смещения и ограничения.
        """
        result = await db_session.execute(query)

        result = result.fetchone()
        if result:
            return result[0]
        return False
