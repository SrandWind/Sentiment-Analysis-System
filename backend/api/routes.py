# -*- coding: utf-8 -*-
"""
API routes for Sentiment Analysis System
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import json
import os
import io

from database.db import get_db, SessionLocal
from database.models import InferenceHistory
from services.lmstudio_client import LMStudioClient
from services.parser import format_inference_result, calculate_confidence
from services.evaluator import compute_sample_metrics
from config import InferencePresets
from api.schemas import (
    InferRequest, InferResponse,
    BatchRequest, BatchResponse, BatchItem,
    HistoryResponse, HistoryItem,
    MetricsResponse, CompareResponse, CompareRequest,
    HealthResponse
)

router = APIRouter()

# Version
VERSION = "1.0.0"

# In-memory batch progress store
batch_progress_store: dict = {}

# Possible paths for metrics JSON files (from eval_v2.py output)
METRICS_FILE_PATHS = [
    "./outputs",           # Running from project root: ./outputs
    "../outputs",          # Running from backend/: ../outputs
    "../../outputs",       # Running from backend/api/: ../../outputs
    "./backend/outputs",   # Alternative path
]


def load_metrics_from_file(variant: str) -> Optional[dict]:
    """
    Load metrics from eval_v2.py generated JSON file.
    Returns None if file does not exist.
    """
    # Try both naming conventions: metrics_{variant}.json and metrics_{variant}_async.json
    filenames = [f"metrics_{variant}.json", f"metrics_{variant}_async.json"]
    for base_path in METRICS_FILE_PATHS:
        for filename in filenames:
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"[info] Loaded metrics from {filepath}")
                    return data
                except Exception as e:
                    print(f"[warn] Failed to load metrics from {filepath}: {e}")
    return None


def get_metrics_for_variant(variant: str) -> dict:
    """
    Get metrics for a model variant.
    Priority: 1) File from eval_v2.py, 2) Hardcoded fallback
    """
    # Try to load from file first
    file_data = load_metrics_from_file(variant)
    if file_data is not None:
        # Ensure model_variant is set
        file_data["model_variant"] = variant
        return file_data

    # Fallback to hardcoded metrics
    if variant in PRECOMPUTED_METRICS:
        return PRECOMPUTED_METRICS[variant].copy()

    return None


# Pre-computed metrics for model comparison (fallback when eval_v2.py files don't exist)
PRECOMPUTED_METRICS = {
    "base": {
        "model_variant": "base",
        "emotion_macro_mae": 0.185,
        "emotion_macro_mse": 0.052,
        "primary_cls_accuracy": 0.72,
        "primary_cls_macro_f1": 0.71,
        "mbti_accuracy": 0.45,
        "mbti_macro_f1": 0.42,
        "json_parse_rate": 0.89,
        "cot7_complete_rate": 0.85,
        "emotion_per_dim_mae": {
            "angry": 0.15, "fear": 0.22, "happy": 0.12,
            "neutral": 0.18, "sad": 0.20, "surprise": 0.24
        },
        "primary_cls_per_class_f1": {
            "angry": 0.75, "fear": 0.65, "happy": 0.82,
            "neutral": 0.68, "sad": 0.72, "surprise": 0.64
        }
    },
    "lora_merged": {
        "model_variant": "lora_merged",
        "emotion_macro_mae": 0.142,
        "emotion_macro_mse": 0.035,
        "primary_cls_accuracy": 0.81,
        "primary_cls_macro_f1": 0.80,
        "mbti_accuracy": 0.52,
        "mbti_macro_f1": 0.48,
        "json_parse_rate": 0.94,
        "cot7_complete_rate": 0.91,
        "emotion_per_dim_mae": {
            "angry": 0.12, "fear": 0.16, "happy": 0.10,
            "neutral": 0.14, "sad": 0.15, "surprise": 0.18
        },
        "primary_cls_per_class_f1": {
            "angry": 0.82, "fear": 0.75, "happy": 0.88,
            "neutral": 0.78, "sad": 0.80, "surprise": 0.72
        }
    },
    "gguf4bit": {
        "model_variant": "gguf4bit",
        "emotion_macro_mae": 0.148,
        "emotion_macro_mse": 0.038,
        "primary_cls_accuracy": 0.79,
        "primary_cls_macro_f1": 0.78,
        "mbti_accuracy": 0.50,
        "mbti_macro_f1": 0.46,
        "json_parse_rate": 0.92,
        "cot7_complete_rate": 0.88,
        "emotion_per_dim_mae": {
            "angry": 0.13, "fear": 0.17, "happy": 0.11,
            "neutral": 0.15, "sad": 0.16, "surprise": 0.19
        },
        "primary_cls_per_class_f1": {
            "angry": 0.80, "fear": 0.73, "happy": 0.86,
            "neutral": 0.76, "sad": 0.78, "surprise": 0.70
        }
    }
}


@router.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest, db: Session = Depends(get_db)):
    """
    Perform single text inference.

    Sends text to LMStudio API and returns structured emotion analysis.
    Supports different inference presets: quick, standard, deep.
    """
    client = LMStudioClient()
    
    # Get preset parameters, allow override via request
    preset = InferencePresets.get_preset(request.preset or "standard")
    max_tokens = request.max_tokens or preset["max_tokens"]
    temperature = request.temperature if request.temperature is not None else preset["temperature"]
    top_p = request.top_p or preset["top_p"]
    repeat_penalty = request.repeat_penalty or preset["repeat_penalty"]

    try:
        # Call LMStudio API with preset parameters
        result = await client.infer(
            prompt=request.text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )

        # Parse output
        parsed = format_inference_result(
            output=result["output"],
            latency_ms=result["latency_ms"],
            model_variant=request.model_variant
        )

        # Save to history
        display_scores = parsed.get("target_scores") or parsed.get("scores", {})
        db_entry = InferenceHistory(
            text=request.text,
            output=result["output"],
            parsed_result=parsed,
            emotion_angry=display_scores.get("angry", 0.0),
            emotion_fear=display_scores.get("fear", 0.0),
            emotion_happy=display_scores.get("happy", 0.0),
            emotion_neutral=display_scores.get("neutral", 0.0),
            emotion_sad=display_scores.get("sad", 0.0),
            emotion_surprise=display_scores.get("surprise", 0.0),
            primary_emotion=parsed["primary_emotion"],
            mbti_type=parsed.get("mbti_type", ""),
            cot_reasoning=parsed["cot"],
            model_variant=request.model_variant,
            latency_ms=parsed["latency_ms"],
            json_parse_ok=parsed["json_parse_ok"],
            cot_complete=parsed["cot_complete"]
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)

        return InferResponse(
            success=True,
            text=request.text,
            output=result["output"],
            scores=display_scores,
            target_scores=parsed.get("target_scores"),
            cot=parsed["cot"],
            primary_emotion=parsed["primary_emotion"],
            mbti_type=parsed.get("mbti_type", ""),
            confidence=parsed.get("confidence", 0.0),
            json_parse_ok=parsed["json_parse_ok"],
            cot_complete=parsed["cot_complete"],
            latency_ms=parsed["latency_ms"],
            model_variant=request.model_variant,
            vad_dimensions=parsed.get("vad_dimensions"),
            emotion_cause=parsed.get("emotion_cause"),
            uncertainty_level=parsed.get("uncertainty_level")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/infer/stream")
async def infer_stream(request: InferRequest):
    """
    Perform streaming inference on text.

    Returns Server-Sent Events (SSE) stream with progressive CoT reasoning.
    Supports different inference presets: quick, standard, deep.
    Saves the final result to history after streaming completes.
    """
    client = LMStudioClient()
    
    # Get preset parameters, allow override via request
    preset = InferencePresets.get_preset(request.preset or "standard")
    max_tokens = request.max_tokens or preset["max_tokens"]
    temperature = request.temperature if request.temperature is not None else preset["temperature"]
    top_p = request.top_p or preset["top_p"]
    repeat_penalty = request.repeat_penalty or preset["repeat_penalty"]

    # Get database session directly for this request
    db = SessionLocal()

    async def generate_stream():
        final_output = None
        final_latency = 0
        parsed_result = None
        try:
            async for chunk in client.infer_stream(
                prompt=request.text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            ):
                final_output = chunk.get("output", "")
                final_latency = chunk.get("latency_ms", 0)
                # Forward all chunks to frontend
                # Special chunks (done=True) include parsed data from lmstudio_client
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"
        finally:
            # Save to database (final_chunk already sent by lmstudio_client)
            if final_output:
                try:
                    # Re-parse for database storage
                    parsed_result = format_inference_result(
                        output=final_output,
                        latency_ms=final_latency,
                        model_variant=request.model_variant
                    )
                    # Use target_scores (CoT-adjusted) for emotion columns
                    display_scores = parsed_result.get("target_scores") or parsed_result.get("scores", {})
                    db_entry = InferenceHistory(
                        text=request.text,
                        output=final_output,
                        parsed_result=parsed_result,
                        emotion_angry=display_scores.get("angry", 0.0),
                        emotion_fear=display_scores.get("fear", 0.0),
                        emotion_happy=display_scores.get("happy", 0.0),
                        emotion_neutral=display_scores.get("neutral", 0.0),
                        emotion_sad=display_scores.get("sad", 0.0),
                        emotion_surprise=display_scores.get("surprise", 0.0),
                        primary_emotion=parsed_result["primary_emotion"],
                        mbti_type=parsed_result.get("mbti_type", ""),
                        cot_reasoning=parsed_result["cot"],
                        model_variant=request.model_variant,
                        latency_ms=parsed_result["latency_ms"],
                        json_parse_ok=parsed_result["json_parse_ok"],
                        cot_complete=parsed_result["cot_complete"]
                    )
                    db.add(db_entry)
                    db.commit()
                except Exception as db_err:
                    print(f"Failed to save streaming history: {db_err}")
                    db.rollback()
                finally:
                    db.close()

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def process_batch_async(batch_id: str, texts: List[str], model_variant: str, preset: dict):
    """Process batch in background and update progress store."""
    client = LMStudioClient()
    
    for i, text in enumerate(texts):
        try:
            result = await client.infer(
                prompt=text,
                max_tokens=preset["max_tokens"],
                temperature=preset["temperature"],
                top_p=preset["top_p"],
                repeat_penalty=preset["repeat_penalty"],
            )
            parsed = format_inference_result(
                output=result["output"],
                latency_ms=result["latency_ms"],
                model_variant=model_variant
            )

            batch_progress_store[batch_id]["results"].append({
                "id": i, "text": text, "success": True, "result": parsed
            })
        except Exception as e:
            batch_progress_store[batch_id]["results"].append({
                "id": i, "text": text, "success": False, "error": str(e)
            })
        
        batch_progress_store[batch_id]["processed"] = i + 1
        batch_progress_store[batch_id]["current_text"] = text[:50]
    
    batch_progress_store[batch_id]["status"] = "completed"
    batch_progress_store[batch_id]["current_text"] = ""
    
    import asyncio
    asyncio.create_task(cleanup_progress(batch_id))


@router.post("/batch/start")
async def batch_infer_start(request: BatchRequest):
    """
    Start batch inference and return batch_id immediately.
    Use GET /batch/progress/{batch_id} to poll for progress.
    Use GET /batch/results/{batch_id} to get final results.
    """
    import uuid
    batch_id = str(uuid.uuid4())[:8]
    
    preset = InferencePresets.get_preset("quick") if request.use_quick_preset else InferencePresets.get_preset("standard")
    
    batch_progress_store[batch_id] = {
        "total": len(request.texts),
        "processed": 0,
        "results": [],
        "status": "processing",
        "current_text": "",
    }
    
    import asyncio
    asyncio.create_task(process_batch_async(batch_id, request.texts, request.model_variant or "base", preset))

    return {"batch_id": batch_id, "total": len(request.texts)}


@router.post("/batch", response_model=BatchResponse)
async def batch_infer(request: BatchRequest):
    """
    Perform batch inference on multiple texts (blocking version).
    For real-time progress, use POST /batch/start instead.
    """
    import uuid
    batch_id = str(uuid.uuid4())[:8]
    
    client = LMStudioClient()
    
    preset = InferencePresets.get_preset("quick") if request.use_quick_preset else InferencePresets.get_preset("standard")
    
    batch_progress_store[batch_id] = {
        "total": len(request.texts),
        "processed": 0,
        "results": [],
        "status": "processing",
    }
    
    results = []
    total = len(request.texts)

    for i, text in enumerate(request.texts):
        try:
            result = await client.infer(
                prompt=text,
                max_tokens=preset["max_tokens"],
                temperature=preset["temperature"],
                top_p=preset["top_p"],
                repeat_penalty=preset["repeat_penalty"],
            )
            parsed = format_inference_result(
                output=result["output"],
                latency_ms=result["latency_ms"],
                model_variant=request.model_variant
            )

            batch_item = BatchItem(
                id=i,
                text=text,
                success=True,
                result=parsed
            )
            results.append(batch_item)
            batch_progress_store[batch_id]["results"].append({
                "id": i, "text": text, "success": True, "result": parsed
            })
        except Exception as e:
            batch_item = BatchItem(
                id=i,
                text=text,
                success=False,
                error=str(e)
            )
            results.append(batch_item)
            batch_progress_store[batch_id]["results"].append({
                "id": i, "text": text, "success": False, "error": str(e)
            })
        
        batch_progress_store[batch_id]["processed"] = i + 1
    
    success_count = sum(1 for r in results if r.success)
    
    batch_progress_store[batch_id]["status"] = "completed"
    
    import asyncio
    asyncio.create_task(cleanup_progress(batch_id))

    return BatchResponse(
        total=len(results),
        success=success_count,
        failed=len(results) - success_count,
        results=results
    )


async def cleanup_progress(batch_id: str):
    """Clean up batch progress after timeout."""
    import asyncio
    await asyncio.sleep(300)  # 5 minutes
    batch_progress_store.pop(batch_id, None)


@router.get("/batch/progress/{batch_id}")
async def get_batch_progress(batch_id: str):
    """
    Get progress of a batch inference by batch_id.
    Returns current progress status with current_text being processed.
    """
    if batch_id not in batch_progress_store:
        return {
            "status": "not_found",
            "message": "Batch not found or already cleaned up"
        }
    
    progress = batch_progress_store[batch_id]
    return {
        "status": progress["status"],
        "total": progress["total"],
        "processed": progress["processed"],
        "progress_percent": int((progress["processed"] / progress["total"]) * 100) if progress["total"] > 0 else 0,
        "current_text": progress.get("current_text", ""),
    }


@router.get("/batch/results/{batch_id}")
async def get_batch_results(batch_id: str):
    """
    Get final results of a completed batch inference.
    """
    if batch_id not in batch_progress_store:
        return {
            "status": "not_found",
            "message": "Batch not found or already cleaned up"
        }
    
    progress = batch_progress_store[batch_id]
    if progress["status"] != "completed":
        return {
            "status": progress["status"],
            "message": "Batch not yet completed",
            "results": None
        }
    
    return {
        "status": "completed",
        "total": progress["total"],
        "success": sum(1 for r in progress["results"] if r.get("success")),
        "failed": sum(1 for r in progress["results"] if not r.get("success")),
        "results": progress["results"]
    }


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get inference history.
    """
    query = db.query(InferenceHistory).order_by(InferenceHistory.created_at.desc())
    total = query.count()
    items = query.offset(offset).limit(limit).all()

    def get_scores(item: InferenceHistory) -> dict:
        """Get emotion scores - prefer target_scores from parsed_result."""
        # Try target_scores first (CoT-adjusted scores)
        if item.parsed_result:
            try:
                target = item.parsed_result.get("target_scores", {})
                if target and any(target.values()):
                    return target
            except (ValueError, TypeError):
                pass
        
        # Fallback: use emotion_analysis or original emotion columns
        emotion_map = {
            "angry": item.emotion_angry,
            "fear": item.emotion_fear,
            "happy": item.emotion_happy,
            "neutral": item.emotion_neutral,
            "sad": item.emotion_sad,
            "surprise": item.emotion_surprise,
        }
        return emotion_map

    def get_confidence(item: InferenceHistory) -> float:
        """Get confidence from parsed_result."""
        if item.parsed_result and "confidence" in item.parsed_result:
            return float(item.parsed_result["confidence"])
        return 0.0

    def get_vad_dimensions(item: InferenceHistory) -> Optional[Dict[str, float]]:
        """Get VAD dimensions from parsed_result."""
        if item.parsed_result and "vad_dimensions" in item.parsed_result:
            return item.parsed_result["vad_dimensions"]
        return None

    def get_uncertainty_level(item: InferenceHistory) -> Optional[str]:
        """Get uncertainty level from parsed_result."""
        if item.parsed_result and "uncertainty_level" in item.parsed_result:
            return item.parsed_result["uncertainty_level"]
        return None

    return HistoryResponse(
        total=total,
        items=[
            HistoryItem(
                id=item.id,
                text=item.text,
                primary_emotion=item.primary_emotion,
                mbti_type=item.mbti_type,
                target_scores=get_scores(item),
                confidence=get_confidence(item),
                latency_ms=item.latency_ms,
                json_parse_ok=item.json_parse_ok,
                created_at=item.created_at,
                vad_dimensions=get_vad_dimensions(item),
                uncertainty_level=get_uncertainty_level(item)
            )
            for item in items
        ]
    )


@router.get("/history/{item_id}")
async def get_history_item(item_id: int, db: Session = Depends(get_db)):
    """
    Get detailed history item.
    """
    item = db.query(InferenceHistory).filter(InferenceHistory.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    def get_display_scores() -> dict:
        """Get display scores - prefer target_scores from parsed_result."""
        if item.parsed_result:
            try:
                target = item.parsed_result.get("target_scores", {})
                if target and any(target.values()):
                    return target
            except (ValueError, TypeError):
                pass
        
        return {
            "angry": item.emotion_angry,
            "fear": item.emotion_fear,
            "happy": item.emotion_happy,
            "neutral": item.emotion_neutral,
            "sad": item.emotion_sad,
            "surprise": item.emotion_surprise
        }

    def get_confidence() -> float:
        """Get confidence from parsed_result."""
        if item.parsed_result and "confidence" in item.parsed_result:
            return float(item.parsed_result["confidence"])
        return 0.0

    def get_vad_dimensions() -> Optional[Dict[str, float]]:
        """Get VAD dimensions from parsed_result."""
        if item.parsed_result and "vad_dimensions" in item.parsed_result:
            return item.parsed_result["vad_dimensions"]
        return None

    def get_uncertainty_level() -> Optional[str]:
        """Get uncertainty level from parsed_result."""
        if item.parsed_result and "uncertainty_level" in item.parsed_result:
            return item.parsed_result["uncertainty_level"]
        return None

    return {
        "id": item.id,
        "text": item.text,
        "output": item.output,
        "parsed_result": item.parsed_result,
        "scores": get_display_scores(),
        "cot": item.cot_reasoning,
        "primary_emotion": item.primary_emotion,
        "mbti_type": item.mbti_type,
        "confidence": get_confidence(),
        "latency_ms": item.latency_ms,
        "json_parse_ok": item.json_parse_ok,
        "cot_complete": item.cot_complete,
        "created_at": item.created_at,
        "vad_dimensions": get_vad_dimensions(),
        "uncertainty_level": get_uncertainty_level()
    }


@router.get("/metrics/{model_variant}", response_model=MetricsResponse)
async def get_metrics(model_variant: str):
    """
    Get pre-computed metrics for a model variant.
    Priority: 1) File from eval_v2.py, 2) Hardcoded fallback
    """
    metrics = get_metrics_for_variant(model_variant)
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model variant '{model_variant}' not found. Available: base, lora_merged, gguf4bit"
        )

    return metrics


@router.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest = None):
    """
    Compare multiple model variants.
    Priority: 1) File from eval_v2.py, 2) Hardcoded fallback
    """
    if request is None:
        request = CompareRequest()

    variants = request.model_variants or ["base", "lora_merged", "gguf4bit"]
    models = []
    for v in variants:
        metrics = get_metrics_for_variant(v)
        if metrics is not None:
            models.append(MetricsResponse(**metrics))

    if not models:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for variants: {variants}"
        )

    # Generate markdown comparison table
    headers = ["模型", "MAE", "分类准确率", "F1", "MBTI 准确率", "JSON 解析率", "CoT 完成率"]
    rows = [headers, ["---"] * len(headers)]
    for m in models:
        rows.append([
            m.model_variant,
            f"{m.emotion_macro_mae:.4f}",
            f"{m.primary_cls_accuracy:.4f}",
            f"{m.primary_cls_macro_f1:.4f}",
            f"{m.mbti_accuracy:.4f}",
            f"{m.json_parse_rate:.4f}",
            f"{m.cot7_complete_rate:.4f}"
        ])

    table = "\n".join("| " + " | ".join(row) + " |" for row in rows)

    return CompareResponse(
        models=models,
        comparison_table=table
    )


# Possible paths for SwanLab CSV files
SWANLAB_CSV_PATHS = [
    "./outputs/swanlab",       # Running from project root
    "../outputs/swanlab",      # Running from backend/
    "../../outputs/swanlab",   # Running from backend/api/
    "./backend/outputs/swanlab", # Alternative path
]


def parse_swanlab_csv(filepath: str) -> List[dict]:
    """Parse SwanLab CSV file and return list of {step, value}"""
    if not os.path.exists(filepath):
        return []
    result = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return []
        header = lines[0].strip()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    step = int(parts[0])
                    value = float(parts[1])
                    result.append({'step': step, 'value': value})
                except ValueError:
                    continue
    return result


@router.get("/training-history")
async def get_training_history():
    """
    Get training history from SwanLab CSV files.
    Returns merged data with train/val loss and accuracy per evaluation step.
    """
    swanlab_path = None
    for base_path in SWANLAB_CSV_PATHS:
        if os.path.exists(base_path):
            swanlab_path = base_path
            break
    
    if not swanlab_path:
        return {"error": "SwanLab data not found", "data": []}
    
    train_loss = parse_swanlab_csv(os.path.join(swanlab_path, "train_loss.csv"))
    train_acc = parse_swanlab_csv(os.path.join(swanlab_path, "train_accuracy.csv"))
    val_loss = parse_swanlab_csv(os.path.join(swanlab_path, "val_loss.csv"))
    val_acc = parse_swanlab_csv(os.path.join(swanlab_path, "val_accuracy.csv"))
    
    train_loss_map = {d['step']: d['value'] for d in train_loss}
    train_acc_map = {d['step']: d['value'] for d in train_acc}
    val_loss_map = {d['step']: d['value'] for d in val_loss}
    val_acc_map = {d['step']: d['value'] for d in val_acc}
    
    all_steps = sorted(set(val_loss_map.keys()))
    
    result = []
    for step in all_steps:
        result.append({
            'step': step,
            'trainLoss': train_loss_map.get(step, 0),
            'valLoss': val_loss_map.get(step, 0),
            'trainAcc': train_acc_map.get(step, 0),
            'valAcc': val_acc_map.get(step, 0),
        })
    
    return {"data": result, "totalSteps": len(result)}


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    """
    lmstudio_client = LMStudioClient()
    lmstudio_ok = await lmstudio_client.health_check()

    db_ok = False
    try:
        db.execute("SELECT 1")
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if (lmstudio_ok and db_ok) else "degraded",
        lmstudio_connected=lmstudio_ok,
        database_connected=db_ok,
        version=VERSION
    )
