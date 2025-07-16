import os
import shutil
import uuid
import json
from zipfile import ZipFile, is_zipfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

import asyncpg
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings, DATE_OF_PHOTOS
from app import crud, ml_logic, schemas
from app.database import create_pool, get_db

# --- Создание папок при старте ---
for dir_path in [
    settings.UPLOADS_DIR,
    settings.PASSPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- Инициализация ML моделей при старте ---
classifier_model = ml_logic.get_classifier_model()
embedder_model = ml_logic.get_embedder_model()


# --- Управление жизненным циклом приложения (пул БД) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Приложение запускается, создаем пул соединений...")
    app.state.pool = await create_pool()
    yield
    print("Приложение останавливается, закрываем пул соединений...")
    await app.state.pool.close()


app = FastAPI(title="Rare Animal Tracker API", lifespan=lifespan, version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def save_upload_file(upload_file: UploadFile, directory: str) -> str:
    """Сохраняет загруженный файл в указанную директорию с уникальным именем."""
    unique_filename = f"{uuid.uuid4().hex[:10]}-{upload_file.filename}"
    file_path = os.path.join(directory, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return unique_filename


async def process_zip_file(zip_path: str) -> list[str]:
    """Распаковывает ZIP-архив и возвращает список имен файлов изображений."""
    extracted_files = []
    with ZipFile(zip_path, "r") as input_zip:
        for member_name in input_zip.namelist():
            if not member_name.startswith('__MACOSX') and member_name.split(".")[-1].lower() in ["png", "jpg", "jpeg"]:
                unique_filename = f"{uuid.uuid4().hex[:10]}-{Path(member_name).name}"
                source = input_zip.open(member_name)
                target_path = Path(settings.UPLOADS_DIR) / unique_filename
                with open(target_path, "wb") as target_file:
                    shutil.copyfileobj(source, target_file)
                extracted_files.append(unique_filename)
    os.remove(zip_path)
    return extracted_files


# --- Эндпоинты API ---

@app.post("/upload/", response_model=schemas.UploadResponse)
async def upload_files_and_process(
        cords_sd: float = Form(...),
        cords_vd: float = Form(...),
        files: list[UploadFile] = File(...),
        conn: asyncpg.Connection = Depends(get_db)
):
    coordinates = f"{cords_sd}, {cords_vd}"
    processed_filenames = []
    for file in files:
        saved_filename = await save_upload_file(file, settings.UPLOADS_DIR)
        saved_filepath = os.path.join(settings.UPLOADS_DIR, saved_filename)
        if is_zipfile(saved_filepath):
            processed_filenames.extend(await process_zip_file(saved_filepath))
        elif file.content_type in ["image/jpeg", "image/png"]:
            processed_filenames.append(saved_filename)
        else:
            os.remove(saved_filepath)

    if not processed_filenames:
        raise HTTPException(status_code=400, detail="Не найдено валидных файлов для обработки.")

    zip_id = await crud.create_zip_record(conn, DATE_OF_PHOTOS, 0, coordinates)
    rare_animals_found_count = 0
    predictions_for_response = []
    animal_counts_diagram = {}

    passport_embeddings = await crud.get_passport_embeddings(conn)
    output_embeddings = await crud.get_output_embeddings(conn)
    all_known_embeddings = passport_embeddings + output_embeddings

    for filename in processed_filenames:
        image_path = os.path.join(settings.UPLOADS_DIR, filename)
        species = ml_logic.classify_species(classifier_model, image_path)
        print(f"Файл: {filename}, Класс: {species}")

        animal_counts_diagram[species] = animal_counts_diagram.get(species, 0) + 1
        passport_id = None
        embedding_str = None

        if species in settings.RARE_ANIMALS_LIST:
            rare_animals_found_count += 1
            embedding = ml_logic.extract_embedding(embedder_model, image_path)
            embedding_str = json.dumps(embedding) if embedding else None

            if embedding and all_known_embeddings:
                similar = ml_logic.find_most_similar_passports(embedding, all_known_embeddings)
                if similar and similar[0][1] > settings.SIMILARITY_THRESHOLD:
                    passport_id = similar[0][0]
                    cords_id = await crud.create_cords_record(conn, DATE_OF_PHOTOS, coordinates, passport_id)
                    await crud.update_passport_cords(conn, passport_id, cords_id)

        await crud.create_output_record(conn, zip_id, species, filename, 0.99, 0, passport_id, embedding_str)

        p_data = schemas.Prediction(
            IMG=filename, type=species, date=DATE_OF_PHOTOS,
            size="N/A", confidence="N/A", passport=passport_id
        )
        predictions_for_response.append(p_data)

    if rare_animals_found_count > 0:
        await crud.update_zip_rare_count(conn, zip_id, rare_animals_found_count)

    return {"pred": predictions_for_response, "diagram": animal_counts_diagram, "coordinates": coordinates}


@app.post("/upload_passport/", response_model=schemas.PassportInDB)
async def upload_passport(
        age: int = Form(...),
        gender: str = Form(...),
        name: str = Form(...),
        cords_sd: float = Form(...),
        cords_vd: float = Form(...),
        image_preview: UploadFile = File(...),
        conn: asyncpg.Connection = Depends(get_db)
):
    if image_preview.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Неверный тип файла изображения.")

    coordinates = f"{cords_sd}, {cords_vd}"
    passport_photo_filename = await save_upload_file(image_preview, settings.PASSPORTS_DIR)
    passport_photo_path = os.path.join(settings.PASSPORTS_DIR, passport_photo_filename)

    species = ml_logic.classify_species(classifier_model, passport_photo_path)
    if species not in settings.RARE_ANIMALS_LIST:
        os.remove(passport_photo_path)
        raise HTTPException(status_code=400, detail=f"На фото определен вид '{species}', который не является редким.")

    embedding = ml_logic.extract_embedding(embedder_model, passport_photo_path)
    if not embedding:
        os.remove(passport_photo_path)
        raise HTTPException(status_code=500, detail="Не удалось извлечь эмбеддинг из фото.")

    embedding_str = json.dumps(embedding)
    passport_id = await crud.create_passport_record(conn, passport_photo_filename, species, age, gender, name,
                                                    embedding_str)
    cords_id = await crud.create_cords_record(conn, DATE_OF_PHOTOS, coordinates, passport_id)
    await crud.update_passport_cords(conn, passport_id, cords_id)

    new_passport_record = await crud.get_passport_by_id(conn, passport_id)
    if new_passport_record:
        return dict(new_passport_record)
    raise HTTPException(status_code=404, detail="Не удалось найти только что созданный паспорт.")


@app.get("/all_passports", response_model=List[schemas.PassportInDB])
async def get_all_passports(conn: asyncpg.Connection = Depends(get_db)):
    passports_records = await crud.get_all_passports(conn)
    return [dict(p) for p in passports_records]


@app.get("/get_passport/{passport_id}", response_model=schemas.PassportInDB)
async def get_passport(passport_id: int, conn: asyncpg.Connection = Depends(get_db)):
    passport_record = await crud.get_passport_by_id(conn, passport_id)
    if not passport_record:
        raise HTTPException(status_code=404, detail="Паспорт не найден")
    return dict(passport_record)


@app.get("/all_zips", response_model=List[schemas.ZipRecord])
async def get_all_zips(conn: asyncpg.Connection = Depends(get_db)):
    zips_records = await crud.get_all_zips(conn)
    return [dict(z) for z in zips_records]


@app.get("/get_uploads/{zip_id}", response_model=schemas.UploadResponse)
async def get_uploads_by_zip(zip_id: int, conn: asyncpg.Connection = Depends(get_db)):
    output_records = await crud.get_output_by_zip_id(conn, zip_id)
    if not output_records:
        raise HTTPException(status_code=404, detail="Записи для данной загрузки не найдены")

    predictions_for_response = []
    animal_counts_diagram = {}

    for record in output_records:
        species = record['species']
        animal_counts_diagram[species] = animal_counts_diagram.get(species, 0) + 1
        p_data = schemas.Prediction(
            IMG=record['processed_photo'],
            type=species,
            date=record['upload_date'].strftime('%Y-%m-%d'),
            size=str(record.get('size', 'N/A')),
            confidence=str(record.get('confidence', 'N/A')),
            passport=record['pass_id']
        )
        predictions_for_response.append(p_data)

    return {"pred": predictions_for_response, "diagram": animal_counts_diagram}


@app.get("/passport/{passport_id}/history")
async def get_passport_history(passport_id: int, conn: asyncpg.Connection = Depends(get_db)):
    history_records = await crud.get_cords_history_by_passport_id(conn, passport_id)
    return [dict(r) for r in history_records]


@app.get("/passport/{passport_id}/photos")
async def get_passport_photos(passport_id: int, conn: asyncpg.Connection = Depends(get_db)):
    photo_records = await crud.get_photos_by_passport_id(conn, passport_id)
    return [record['processed_photo'] for record in photo_records]


@app.post("/assign_passport/")
async def assign_passport_to_photo(
        image_name: str = Form(...),
        passport_id: int = Form(...),
        cords_sd: float = Form(...),
        cords_vd: float = Form(...),
        conn: asyncpg.Connection = Depends(get_db)
):
    source_path = os.path.join(settings.UPLOADS_DIR, image_name)
    if not os.path.isfile(source_path):
        raise HTTPException(status_code=404, detail="Фото для присвоения не найдено.")

    embedding = ml_logic.extract_embedding(embedder_model, source_path)
    embedding_str = json.dumps(embedding) if embedding else None

    await crud.assign_photo_to_passport(conn, image_name, passport_id, embedding_str)

    coordinates = f"{cords_sd}, {cords_vd}"
    cords_id = await crud.create_cords_record(conn, DATE_OF_PHOTOS, coordinates, passport_id)
    await crud.update_passport_cords(conn, passport_id, cords_id)

    return {"message": "Фото успешно присвоено паспорту.", "passport_id": passport_id}


@app.get("/image/passport/{image_name}")
async def get_passport_image(image_name: str):
    passport_image_path = os.path.join(settings.PASSPORTS_DIR, image_name)
    if os.path.isfile(passport_image_path):
        return FileResponse(passport_image_path)

    upload_image_path = os.path.join(settings.UPLOADS_DIR, image_name)
    if os.path.isfile(upload_image_path):
        return FileResponse(upload_image_path)

    raise HTTPException(status_code=404, detail="Изображение не найдено")


@app.post("/create_passport_from_upload/", response_model=schemas.PassportInDB)
async def create_passport_from_upload(
        image_name: str = Form(...),
        age: int = Form(...),
        gender: str = Form(...),
        name: str = Form(...),
        cords_sd: float = Form(...),
        cords_vd: float = Form(...),
        conn: asyncpg.Connection = Depends(get_db)
):
    source_path = os.path.join(settings.UPLOADS_DIR, image_name)
    if not os.path.isfile(source_path):
        raise HTTPException(status_code=404, detail="Исходное фото для создания паспорта не найдено.")

    passport_photo_path = os.path.join(settings.PASSPORTS_DIR, image_name)
    shutil.copy(source_path, passport_photo_path)

    output_record = await conn.fetchrow("SELECT species FROM Outputs WHERE processed_photo = $1", image_name)
    if not output_record:
        raise HTTPException(status_code=404, detail="Запись о фото не найдена в базе данных.")
    species = output_record['species']

    embedding = ml_logic.extract_embedding(embedder_model, passport_photo_path)
    if not embedding:
        raise HTTPException(status_code=500, detail="Не удалось извлечь эмбеддинг из фото.")

    embedding_str = json.dumps(embedding)

    passport_id = await crud.create_passport_record(
        conn, image_name, species, age, gender, name, embedding_str
    )

    # --- ИЗМЕНЕНИЕ: Заменяем прямой вызов на функцию из crud ---
    await crud.update_output_with_passport_and_embedding(
        conn=conn,
        passport_id=passport_id,
        embedding_str=embedding_str,
        image_name=image_name
    )

    coordinates = f"{cords_sd}, {cords_vd}"
    cords_id = await crud.create_cords_record(conn, DATE_OF_PHOTOS, coordinates, passport_id)
    await crud.update_passport_cords(conn, passport_id, cords_id)

    new_passport_record = await crud.get_passport_by_id(conn, passport_id)
    if new_passport_record:
        return dict(new_passport_record)
    raise HTTPException(status_code=500, detail="Не удалось создать или найти паспорт.")