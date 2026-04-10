# -*- coding: utf-8 -*-
"""
SQLAlchemy models for database tables
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


def utc_now():
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class InferenceHistory(Base):
    """Store inference history for replay and analysis."""
    __tablename__ = "inference_history"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)  # Input text
    output = Column(Text)  # Raw model output
    parsed_result = Column(JSON)  # Parsed JSON result

    # Emotion scores
    emotion_angry = Column(Float, default=0.0)
    emotion_fear = Column(Float, default=0.0)
    emotion_happy = Column(Float, default=0.0)
    emotion_neutral = Column(Float, default=0.0)
    emotion_sad = Column(Float, default=0.0)
    emotion_surprise = Column(Float, default=0.0)

    # Classification results
    primary_emotion = Column(String(50))
    mbti_type = Column(String(10))

    # CoT reasoning (stored as JSON for structured access)
    cot_reasoning = Column(JSON)

    # Metadata
    model_variant = Column(String(50), default="gguf4bit")  # base/lora_merged/gguf4bit
    latency_ms = Column(Float)  # Inference latency
    json_parse_ok = Column(Boolean, default=False)  # Whether output was valid JSON
    cot_complete = Column(Boolean, default=False)  # Whether all 7 CoT steps present

    created_at = Column(DateTime(timezone=True), default=utc_now)
