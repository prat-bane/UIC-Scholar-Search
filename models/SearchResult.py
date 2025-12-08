from pydantic import BaseModel

class SearchResult(BaseModel):
    professor: str
    paper_title: str
    matched_section: str
    snippet: str
    score: float
    source_type: str
 
