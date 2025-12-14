from typing import List
from pydantic import BaseModel, Field

class ClassificationRequestBatch(BaseModel):
    texts: List[str] = Field(...,
                             min_length=1,
                             max_length=8,
                             description="Список текстов для классификации", 
                             example=['Привет, я твой друг, я тебя не обижу', 'Привет! Я твой друг! Я тебя не обижу!'])


class ClassificationResponseBatch(BaseModel):
    predicted_labels: List[str]