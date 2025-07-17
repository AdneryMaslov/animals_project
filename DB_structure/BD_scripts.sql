-- Таблица для хранения информации о загрузках (ZIP-архивах или фото)
CREATE TABLE Zips (
    id SERIAL PRIMARY KEY,
    upload_date DATE NOT NULL,
    rare_animals_count INTEGER NOT NULL,
    coordinates TEXT
);

-- Таблица для паспортов редких животных
CREATE TABLE Passports (
    id SERIAL PRIMARY KEY,
    image_preview TEXT,
    type TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    cords_id INTEGER,
    embanding TEXT NOT NULL,
    name TEXT
);

-- Таблица для хранения информации о местоположениях
CREATE TABLE Cords (
    id SERIAL PRIMARY KEY,
    passport_id INTEGER REFERENCES Passports(id) ON DELETE SET NULL, -- Связь с паспортом
    date DATE NOT NULL,
    coordinates TEXT NOT NULL
);


-- Добавляем внешнюю связь для Passports после создания Cords
ALTER TABLE Passports ADD CONSTRAINT fk_cords
FOREIGN KEY (cords_id) REFERENCES Cords(id) ON DELETE SET NULL;


-- Основная таблица для хранения результатов распознавания
CREATE TABLE Outputs (
    id SERIAL PRIMARY KEY,
    zip_id INTEGER REFERENCES Zips(id) ON DELETE CASCADE,
    species TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    processed_photo TEXT,
    confidence REAL,
    size REAL,
    pass_id INTEGER REFERENCES Passports(id) ON DELETE SET NULL
);

-- Связующая таблица для отношения "многие-ко-многим" между Outputs и Passports
CREATE TABLE Output_Passports (
    output_id INTEGER NOT NULL REFERENCES Outputs(id) ON DELETE CASCADE,
    passport_id INTEGER NOT NULL REFERENCES Passports(id) ON DELETE CASCADE,
    PRIMARY KEY (output_id, passport_id) -- Гарантирует уникальность связи
);
