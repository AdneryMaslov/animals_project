import json
from datetime import datetime
from typing import List, Optional
import asyncpg

# --- Функции чтения (SELECT) ---

async def get_all_passports(conn: asyncpg.Connection):
    """Получает все паспорта из базы данных."""
    return await conn.fetch("SELECT * FROM passports ORDER BY id")

async def get_passport_embeddings(conn: asyncpg.Connection) -> List[tuple[int, List[float]]]:
    """Получает ID и эмбеддинги для всех существующих паспортов."""
    passports = await conn.fetch("SELECT id, embeding FROM passports WHERE embeding IS NOT NULL")
    return [(p['id'], json.loads(p['embeding'])) for p in passports]

async def get_output_embeddings(conn: asyncpg.Connection) -> List[tuple[int, List[float]]]:
    """Получает ID паспорта и эмбеддинги из всех ранее обработанных и присвоенных фото."""
    query = "SELECT pass_id, embedding FROM Outputs WHERE pass_id IS NOT NULL AND embedding IS NOT NULL"
    records = await conn.fetch(query)
    results = []
    for r in records:
        try:
            results.append((r['pass_id'], json.loads(r['embedding'])))
        except (json.JSONDecodeError, TypeError):
            continue
    return results

async def get_passport_by_id(conn: asyncpg.Connection, passport_id: int):
    """Получает один паспорт по его ID."""
    return await conn.fetchrow("SELECT * FROM passports WHERE id = $1", passport_id)

async def get_all_zips(conn: asyncpg.Connection):
    """Получает историю всех загрузок."""
    return await conn.fetch("SELECT * FROM zips ORDER BY upload_date DESC, id DESC")

async def get_output_by_zip_id(conn: asyncpg.Connection, zip_id: int):
    """Получает все записи Outputs для конкретного zip_id, включая дату загрузки."""
    query = """
        SELECT o.*, z.upload_date
        FROM Outputs o
        JOIN Zips z ON o.zip_id = z.id
        WHERE o.zip_id = $1
    """
    return await conn.fetch(query, zip_id)

async def get_cords_history_by_passport_id(conn: asyncpg.Connection, passport_id: int):
    """Получает всю историю координат для одного паспорта."""
    query = "SELECT date, coordinates FROM Cords WHERE passport_id = $1 ORDER BY date DESC"
    return await conn.fetch(query, passport_id)

async def get_photos_by_passport_id(conn: asyncpg.Connection, passport_id: int):
    """Получает все имена файлов фотографий для одного паспорта из таблицы Outputs."""
    query = "SELECT processed_photo FROM Outputs WHERE pass_id = $1 ORDER BY id DESC"
    return await conn.fetch(query, passport_id)

# --- Функции создания (INSERT) ---

async def create_zip_record(conn: asyncpg.Connection, upload_date_str: str, rare_animals_count: int, coordinates: str) -> int:
    """Создает запись о новой сессии загрузки."""
    upload_date = datetime.strptime(upload_date_str, '%Y-%m-%d').date()
    query = 'INSERT INTO Zips (upload_date, rare_animals_count, coordinates) VALUES ($1, $2, $3) RETURNING id;'
    return await conn.fetchval(query, upload_date, rare_animals_count, coordinates)

async def create_output_record(conn: asyncpg.Connection, zip_id: int, species: str, img_name: str, confidence: float, size: float, pass_id: Optional[int], embedding_str: Optional[str] = None) -> int:
    """Создает запись о результате распознавания для одного фото, включая эмбеддинг."""
    query = 'INSERT INTO Outputs (zip_id, species, count, processed_photo, confidence, size, pass_id, embedding) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id;'
    return await conn.fetchval(query, zip_id, species, 1, img_name, confidence, size, pass_id, embedding_str)

async def create_passport_record(conn: asyncpg.Connection, image_preview_path: str, species: str, age: int, gender: str, name: str, embedding_str: str) -> int:
    """Создает запись о новом паспорте для редкого животного."""
    query = 'INSERT INTO Passports (image_preview, type, age, gender, name, embeding) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id;'
    return await conn.fetchval(query, image_preview_path, species, age, gender, name, embedding_str)

async def create_cords_record(conn: asyncpg.Connection, date: str, coordinates: str, pass_id: Optional[int] = None) -> int:
    """Создает запись о местоположении."""
    query = 'INSERT INTO cords (passport_id, date, coordinates) VALUES ($1, $2, $3) RETURNING id'
    parsed_date = datetime.strptime(date, '%Y-%m-%d').date() if date else datetime.now().date()
    return await conn.fetchval(query, pass_id, parsed_date, coordinates)

# --- Функции обновления (UPDATE) ---

async def update_passport_cords(conn: asyncpg.Connection, pass_id: int, cords_id: int):
    """Обновляет ссылку на последние координаты в паспорте животного."""
    query = 'UPDATE passports SET cords_id = $1 WHERE id = $2'
    await conn.execute(query, cords_id, pass_id)

async def update_zip_rare_count(conn: asyncpg.Connection, zip_id: int, count: int):
    """Обновляет счетчик редких животных для записи о загрузке."""
    query = "UPDATE Zips SET rare_animals_count = $1 WHERE id = $2"
    await conn.execute(query, count, zip_id)

async def assign_photo_to_passport(conn: asyncpg.Connection, image_name: str, passport_id: int, embedding_str: Optional[str] = None):
    """Присваивает pass_id и эмбеддинг фотографии в таблице Outputs."""
    query = "UPDATE Outputs SET pass_id = $1, embedding = $2 WHERE processed_photo = $3"
    await conn.execute(query, passport_id, embedding_str, image_name)

async def update_output_with_passport_id(conn: asyncpg.Connection, image_name: str, passport_id: int):
    """Находит запись в Outputs по имени файла и присваивает ей pass_id."""
    query = "UPDATE Outputs SET pass_id = $1 WHERE processed_photo = $2"
    await conn.execute(query, passport_id, image_name)

async def update_output_with_passport_and_embedding(conn: asyncpg.Connection, passport_id: int, embedding_str: str, image_name: str):
    """Обновляет запись в Outputs, устанавливая pass_id и эмбеддинг."""
    query = "UPDATE Outputs SET pass_id = $1, embedding = $2 WHERE processed_photo = $3"
    await conn.execute(query, passport_id, embedding_str, image_name)