# -*- coding: utf-8 -*-
"""
Backend configuration for Sentiment Analysis System
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class InferencePresets:
    """Predefined inference parameter presets for different scenarios."""
    
    # 快速推理：短句/单情绪简单文本（≤50字），基础CoT
    QUICK = {
        "max_tokens": 1536,
        "temperature": 0.03,
        "top_p": 0.75,
        "repeat_penalty": 1.10,
    }
    
    # 标准推理：常规长度文本（50-300字），完整7步CoT【默认推荐档】
    STANDARD = {
        "max_tokens": 2560,
        "temperature": 0.05,
        "top_p": 0.80,
        "repeat_penalty": 1.08,
    }
    
    # 深度推理：长文本/多段落（≥300字），混合情绪/反讽/否定词密集
    DEEP = {
        "max_tokens": 4096,
        "temperature": 0.07,
        "top_p": 0.85,
        "repeat_penalty": 1.05,
    }
    
    @classmethod
    def get_preset(cls, name: str) -> dict:
        """Get preset by name, default to STANDARD."""
        return getattr(cls, name.upper(), cls.STANDARD)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # LMStudio API settings
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "qwen3-8b-instruct"

    # Database settings (default to in-memory SQLite for no-config startup)
    database_url: str = "sqlite+aiosqlite:///:memory:"

    # Deploy mode: local or server
    deploy_mode: str = "local"

    # CORS origins (comma-separated for server mode)
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # Inference settings - Standard preset (default)
    max_tokens: int = 2560
    temperature: float = 0.05
    top_p: float = 0.80
    repeat_penalty: float = 1.08
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Timeout settings
    timeout: float = 180.0
    
    # Default inference preset
    inference_preset: str = "standard"
    
    # Batch inference settings (uses quick preset by default)
    batch_max_tokens: int = 1024
    batch_temperature: float = 0.1
    batch_repeat_penalty: float = 1.15

    @property
    def cors_origin_list(self) -> list:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def standard_preset(self) -> dict:
        """Get standard inference preset."""
        return InferencePresets.get_preset("standard")
    
    @property
    def quick_preset(self) -> dict:
        """Get quick inference preset for batch processing."""
        return InferencePresets.get_preset("quick")
    
    @property
    def deep_preset(self) -> dict:
        """Get deep inference preset for complex analysis."""
        return InferencePresets.get_preset("deep")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars


settings = Settings()
