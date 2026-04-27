from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class Post(BaseModel):
    id: str
    text: str
    category: Optional[str] = None
    confidence: Optional[float] = None
    sentiment: Optional[str] = None
    toxicity_score: float = 0.0
    toxicity_detail: Optional[dict] = None
    embedding: Optional[List[float]] = None
    content_type: str = "text"   # text | image | video
    likes: int = 0
    views: int = 0
    created_at: Optional[datetime] = None

class PostCreate(BaseModel):
    text: str
    content_type: str = "text"
    likes: int = 0
    views: int = 0