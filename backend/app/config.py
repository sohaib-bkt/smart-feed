from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    environment: str = "development"
    firebase_credentials: str = "./serviceAccountKey.json"
    toxicity_threshold: float = 0.20
    max_candidates: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()