# -*- coding: utf-8 -*-
"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


class InferRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze")
    model_variant: Optional[str] = Field("gguf4bit", description="Model variant: base/lora_merged/gguf4bit")
    preset: Optional[str] = Field("standard", description="Inference preset: quick/standard/deep")
    max_tokens: Optional[int] = Field(None, description="Override max tokens")
    temperature: Optional[float] = Field(None, description="Override temperature")
    top_p: Optional[float] = Field(None, description="Override top_p")
    repeat_penalty: Optional[float] = Field(None, description="Override repeat penalty")


class InferResponse(BaseModel):
    success: bool
    text: str
    output: str
    scores: Dict[str, float]
    target_scores: Optional[Dict[str, float]] = None
    cot: Dict[str, str]
    primary_emotion: str
    mbti_type: str
    confidence: float
    json_parse_ok: bool
    cot_complete: bool
    latency_ms: float
    model_variant: str
    vad_dimensions: Optional[Dict[str, float]] = None
    emotion_cause: Optional[str] = None
    uncertainty_level: Optional[str] = "medium"
    risk_warning: Optional[str] = None


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    model_variant: Optional[str] = "gguf4bit"
    output_format: Optional[str] = "json"
    use_quick_preset: Optional[bool] = Field(True, description="Use quick preset for faster batch processing")


class BatchItem(BaseModel):
    id: int
    text: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    total: int
    success: int
    failed: int
    results: List[BatchItem]


class HistoryItem(BaseModel):
    id: int
    text: str
    primary_emotion: str
    mbti_type: Optional[str] = ""
    target_scores: Optional[Dict[str, float]] = None
    confidence: float
    latency_ms: float
    json_parse_ok: bool
    created_at: datetime
    vad_dimensions: Optional[Dict[str, float]] = None
    uncertainty_level: Optional[str] = None


class HistoryResponse(BaseModel):
    total: int
    items: List[HistoryItem]


class MetricsResponse(BaseModel):
    model_variant: str
    emotion_macro_mae: float
    emotion_macro_mse: float
    primary_cls_accuracy: float
    primary_cls_macro_f1: float
    primary_cls_macro_auc: Optional[float] = None
    primary_cls_macro_ap: Optional[float] = None
    mbti_accuracy: float
    mbti_macro_f1: float
    json_parse_rate: float
    cot7_complete_rate: float
    emotion_per_dim_mae: Optional[Dict[str, float]] = None
    emotion_per_dim_mse: Optional[Dict[str, float]] = None
    primary_cls_per_class_f1: Optional[Dict[str, float]] = None
    primary_cls_per_class_metrics: Optional[Dict[str, Any]] = None
    primary_cls_confusion_matrix: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    throughput_sps: Optional[float] = None
    vram_gb: Optional[float] = None


class CompareRequest(BaseModel):
    model_variants: Optional[List[str]] = ["base", "lora_merged", "gguf4bit"]


class CompareResponse(BaseModel):
    models: List[MetricsResponse]
    comparison_table: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    lmstudio_connected: bool
    database_connected: bool
    version: str
