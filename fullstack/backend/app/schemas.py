from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date

class Prediction(BaseModel):
    IMG: str
    type: str
    date: str
    size: str
    confidence: str
    passport: Optional[int] = None

class UploadResponse(BaseModel):
    pred: List[Prediction]
    diagram: dict[str, int]
    coordinates: Optional[str] = None

class PassportBase(BaseModel):
    age: int
    gender: str
    name: str

class PassportCreate(PassportBase):
    cords_sd: float = Field(..., description="Широта")
    cords_vd: float = Field(..., description="Долгота")

class PassportInDB(PassportBase):
    id: int
    image_preview: Optional[str] = None
    type: str

    class Config:
        from_attributes = True

class ZipRecord(BaseModel):
    id: int
    upload_date: date
    rare_animals_count: int
    coordinates: Optional[str] = None

    class Config:
        from_attributes = True

