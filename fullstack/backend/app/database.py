import asyncpg
from fastapi import Request
from config import settings

async def create_pool():
    """
    Создает пул соединений с бд PostgreSQL
    """
    pool = await asyncpg.create_pool(
        user=settings.DB_USER,
        password=settings.DB_PASS,
        database=settings.DB_NAME,
        host=settings.DB_HOST,
        port=settings.DB_PORT
    )
    return pool

async def get_db(request: Request) -> asyncpg.Connection:
    async with request.app.state.pool.acquire() as connection:
        yield connection
