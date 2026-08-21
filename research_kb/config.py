"""Standalone configuration used by the audio module."""

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Validated settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_prefix="FILE_ASSISTANT_",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Field(
        default=Path("data"),
        validation_alias=AliasChoices(
            "VOICETRACE_DATA_DIR",
            "FILE_ASSISTANT_DATA_DIR",
            "RESEARCH_KB_DATA_DIR",
        ),
    )
    audio_model_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "VOICETRACE_AUDIO_MODEL_DIR",
            "VOICE_ASSISTANT_AUDIO_MODEL_DIR",
            "FILE_ASSISTANT_AUDIO_MODEL_DIR",
            "RESEARCH_KB_AUDIO_MODEL_DIR",
        ),
    )
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_timeout_seconds: float = Field(default=60, gt=0)
    deepseek_max_retries: int = Field(default=2, ge=0, le=5)
    qwen_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DASHSCOPE_API_KEY", "FILE_ASSISTANT_QWEN_API_KEY"),
        repr=False,
    )
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "FILE_ASSISTANT_DEEPSEEK_API_KEY"),
        repr=False,
    )
