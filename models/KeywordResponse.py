from pydantic import BaseModel

class KeywordResponse(BaseModel):
    keywords: list[str]