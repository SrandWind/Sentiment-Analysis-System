# -*- coding: utf-8 -*-
"""
FastAPI main application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os

from config import settings
from api.routes import router
from database.db import engine, init_db
from database.models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables (with error handling for no-config mode)
try:
    init_db()
except Exception as e:
    logger.warning(f"Database initialization skipped: {e}")

app = FastAPI(
    title="情感分析系统",
    description="基于多任务 LoRA 微调大模型的社交平台情感分析系统",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve frontend in production (static files)
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend SPA."""
        if full_path.startswith("api"):
            return {"error": "API endpoint not found"}

        index_path = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"error": "Frontend not built. Run 'npm run build' in frontend directory."}


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    print(f"🚀 情感分析系统启动")
    print(f"   模式：{settings.deploy_mode}")
    print(f"   地址：http://{settings.backend_host}:{settings.backend_port}")
    print(f"   LMStudio: {settings.lmstudio_base_url}")
    print(f"   数据库：{settings.database_url}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.deploy_mode == "local"
    )
