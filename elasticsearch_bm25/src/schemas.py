from typing import List, Optional
from pydantic import BaseModel

class HQ(BaseModel):
    id: Optional[int] = None
    comic_name: Optional[str] = None
    issue_title: Optional[str] = None
    issue_description: Optional[str] = None
    penciler: Optional[str] = None
    writer: Optional[str] = None
    cover_artist: Optional[str] = None

class SearchResponse(BaseModel):
    data: List[HQ]