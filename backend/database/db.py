# -*- coding: utf-8 -*-
"""
Database connection and session management
Support both MySQL and SQLite (including in-memory)
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import settings
import logging

logger = logging.getLogger(__name__)

# Parse database URL to determine engine type
db_url = settings.database_url

if db_url.startswith("mysql"):
    # MySQL configuration
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=20,
        max_overflow=10,
        pool_recycle=3600,
        echo=False
    )
elif ":memory:" in db_url:
    # In-memory SQLite (no persistence, for testing)
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    logger.info("Using in-memory database (data will be lost on restart)")
else:
    # File-based SQLite
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")


def get_db():
    """Dependency for getting database session in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
