from pydantic import BaseModel, Field
from typing import Optional, List

class UserData(BaseModel):
    user_id: int = Field(..., description="ID del usuario (1-610)", ge=1, le=610)

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: Optional[float] = None

class RecommendationResponse(BaseModel):
    status: str
    user_cluster: int
    action_taken: str
    action_description: str
    recommendations: List[MovieRecommendation]
    message: Optional[str] = None

class FeedbackData(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0)
    action_taken: int = Field(..., ge=0, le=2)