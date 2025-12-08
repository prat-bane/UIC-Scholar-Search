from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    limit: int = 10