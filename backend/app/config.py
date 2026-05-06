from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    firebase_credentials: str = "./serviceAccountKey.json"
    toxicity_threshold: float = 0.20
    max_candidates: int = 200

settings = Settings()