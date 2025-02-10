from langchain.pydantic_v1 import BaseModel
from typing import List

class ApptainerFilenameTemplateV1(BaseModel):
    filename: str

class ApptainerFilenameAppTemplateV1(BaseModel):
    filename: str
    app: str | None

class ApptainerFilenameListTemplateV1(BaseModel):
    filenames: List[str]