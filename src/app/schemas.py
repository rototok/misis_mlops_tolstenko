from typing import List
from pydantic import BaseModel, Field


class Text(BaseModel):
    text: str = Field(..., example='Привет, я твой друг, я тебя не обижу')


class Texts(BaseModel):
    texts: List[str] = Field(..., example=['Привет, я твой друг, я тебя не обижу', 'Привет! Я твой друг! Я тебя не обижу!'])


class ModelPredictResponse(BaseModel):
    predicted_label: str


class ModelPredictBatchResponse(BaseModel):
    predicted_labels: List[str]